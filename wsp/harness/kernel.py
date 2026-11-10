"""The analysis kernel of experiment C: a persistent Python process the writer's code runs in.

It holds the working copy of a trained network and hands the code a small API (see API_DOC, which
is also what the prompt shows). Weights may be read and written freely; data is only reachable by
running the network, and every such run is counted against a budget of forward passes.

Protocol with the driver (run_edit.py), one JSON object per line:
    driver -> kernel   {"code": "..."}
    kernel -> driver   {"output": "...", "passes_used": n, "seconds": s, "refused": bool}

After every block the working weights and the pass count are saved to --work, so that a kernel
killed for running too long can be restarted where it stood (variables are lost, weights are not).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import guard  # noqa: E402
import netspec  # noqa: E402

MAX_FRAMES = 24000
MARGIN = 1000
OUTPUT_CAP = 9000

API_DOC = """\
  W            mapping: tensor name -> numpy array (a live view of the WORKING network; names and
               shapes as in the tensor table). Assign into it to edit, e.g.
               W["head2.b"][2] += 0.3   or   W["head1.Wf"][:, 7] = 0   or   W["conv.b"] = new_array
               (same shape). Reading and writing W is free.
  BASE         dict with the same names: a read-only copy of the original trained weights.
  halves()     list of (index, source, n_frames) for the practice halves. Free.
  forward(half, start=0, n=12000)
               COSTS 1 PASS. Runs the working network on frames [start, start+n) of practice half
               `half` (original view; n <= 24000; 1000 frames of context are added on each side
               as in the real pipeline) and returns a dict of float32 arrays, one row per frame:
                 "x" (n,48) inputs        "y" (n,) reference labels, 0 = H, 1 = A, 2 = D
                 "a" (n,128)  "h" (n,128)  "g1" (n,256: gru1f | gru1b)  "g2" (n,256: qf | qb)
                 "m" (n,128)  "logits" (n,3)  "logp" (n,3) log-softmax of the logits
  score()      COSTS 1 PASS. Runs the working network on ALL practice halves through the real
               pipeline (four views averaged, decoder) and returns a dict per source ("DFL", "SC",
               "MET", "ALL") with spell_f1, macro_f1, n_pred_spells, n_gt_spells, per_class,
               frame_acc_decoded, frame_acc_argmax, confusion_rows_truth_cols_pred. Takes minutes.
  decode(logp) the decoder alone: (n,3) log-probabilities -> (n,) labels 0/1/2. Free.
  metrics(y, labels)
               spell_f1, macro_f1 and possession counts of a label sequence against reference
               labels. Free. With decode() this lets you test an idea on activations you already
               hold (for example recompute m and the logits from stored g2 and h with changed
               head weights) without spending a pass.
  passes_left()  how many passes remain. Free.
  np, math     numpy and math are already imported."""


class Weights:
    """W in the writer's namespace. W[name] is a live view; W[name] = value writes IN PLACE, so
    `W["head2.b"] = W["head2.b"] + 1` edits the network instead of silently rebinding a key."""

    def __init__(self, state):
        self.views = {a: netspec.view(state, a) for a in netspec.ALIASES}

    def __getitem__(self, name):
        return self.views[name]

    def __setitem__(self, name, value):
        self.views[name][...] = np.asarray(value, np.float32)

    def __contains__(self, name):
        return name in self.views

    def __iter__(self):
        return iter(self.views)

    def __len__(self):
        return len(self.views)

    def keys(self):
        return self.views.keys()

    def items(self):
        return self.views.items()

    def values(self):
        return self.views.values()


class Kernel:
    def __init__(self, base_path, budget, work, split):
        self.work, self.budget, self.split = work, budget, split
        os.makedirs(work, exist_ok=True)
        self.base = netspec.from_checkpoint(base_path)
        saved = os.path.join(work, "working.npz")
        self.state = netspec.load_sparse(saved) if os.path.exists(saved) else netspec.copy_state(self.base)
        meta = os.path.join(work, "passes.json")
        self.used = json.load(open(meta))["used"] if os.path.exists(meta) else 0
        self.log = []                               # what each pass was spent on
        base_views = {}
        for alias in netspec.ALIASES:
            arr = netspec.view(self.base, alias).copy()
            arr.setflags(write=False)
            base_views[alias] = arr
        self.ns = {
            "__builtins__": guard.safe_builtins(), "__name__": "analysis",
            "np": np, "math": __import__("math"),
            "W": Weights(self.state),
            "BASE": base_views,
            "halves": self.halves, "forward": self.forward, "score": self.score,
            "decode": self.decode, "metrics": self.metrics, "passes_left": self.passes_left,
        }

    # -- API ------------------------------------------------------------------------------------
    def passes_left(self):
        return self.budget - self.used

    def _spend(self, what):
        if self.used >= self.budget:
            raise RuntimeError("no forward passes left (budget %d)" % self.budget)
        self.used += 1
        self.log.append(what)

    def halves(self):
        import evaluate
        return [(i, h["src"], h["frames"]) for i, h in enumerate(evaluate.split_index(self.split))]

    def forward(self, half, start=0, n=12000):
        import torch
        import evaluate
        half, start, n = int(half), int(start), int(n)
        index = evaluate.split_index(self.split)
        if not 0 <= half < len(index):
            raise ValueError("half must be in 0..%d" % (len(index) - 1))
        if not 0 < n <= MAX_FRAMES:
            raise ValueError("n must be in 1..%d" % MAX_FRAMES)
        T = index[half]["frames"]
        if not 0 <= start < T:
            raise ValueError("start must be in 0..%d" % (T - 1))
        self._spend("forward(half=%d, start=%d, n=%d)" % (half, start, n))
        data = evaluate.load_half(self.split, half)
        X, y = data["views"][0][0], data["y"]
        stop = min(T, start + n)
        a0, b0 = max(0, start - MARGIN), min(T, stop + MARGIN)
        net = evaluate.net_from_state(self.state)
        gru1 = torch.nn.GRU(netspec.HIDDEN, netspec.HIDDEN, num_layers=1, batch_first=True,
                            bidirectional=True)
        ts = netspec.to_torch_state(self.state)
        gru1.load_state_dict({k[len("gru."):]: v for k, v in ts.items()
                              if k.startswith("gru.") and "_l0" in k})
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(X[a0:b0], np.float32))[None]
            a = net.inp(x)
            h = a + torch.relu(net.conv(a.transpose(1, 2))).transpose(1, 2)
            g1, _ = gru1.eval()(h)
            g2, _ = net.gru(h)
            m = net.head[1](net.head[0](torch.cat([g2, h], -1)))
            logits = net.head[2](m)
            logp = torch.log_softmax(logits, -1)
        cut = slice(start - a0, start - a0 + (stop - start))
        out = {k: v[0, cut].numpy().astype(np.float32) for k, v in
               (("a", a), ("h", h), ("g1", g1), ("g2", g2), ("m", m), ("logits", logits),
                ("logp", logp))}
        out["x"] = np.asarray(X[start:stop], np.float32)
        out["y"] = np.asarray(y[start:stop]).astype(np.int64)
        return out

    def score(self):
        import evaluate
        self._spend("score()")
        result = evaluate.score_state(self.state, self.split)
        return result["sources"]

    def decode(self, logp):
        import evaluate
        from model import viterbi
        logp = np.asarray(logp, np.float64)
        if logp.ndim != 2 or logp.shape[1] != 3:
            raise ValueError("decode expects an (n, 3) array of log-probabilities")
        sh, sa = evaluate.decoder()
        return viterbi(np.maximum(logp, np.log(1e-6)), sh, sa).astype(np.int64)

    def metrics(self, y, labels):
        import possession_metrics as pm
        r = pm.evaluate(np.asarray(y).astype(np.int8), np.asarray(labels).astype(np.int8))
        return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
                "n_gt_spells": int(r["n_gt_spells"]), "n_pred_spells": int(r["n_pred_spells"])}

    # -- running a block ------------------------------------------------------------------------
    def run(self, code):
        t0 = time.time()
        buf = io.StringIO()
        refused = False
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                compiled = guard.check(code)
                exec(compiled, self.ns)
            except guard.Refused as exc:
                refused = True
                print("REFUSED, nothing was run: %s" % exc)
            except BaseException:                       # report, never die on the writer's error
                lines = traceback.format_exc().splitlines()
                keep = [l for l in lines if "<analysis>" in l or not l.startswith("  ")]
                print("\n".join(keep[-12:]))
        # the working network must stay a finite float32 network of the right shapes
        for k, v in self.state.items():
            if not np.all(np.isfinite(v)):
                v[...] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                buf.write("\n[kernel] non-finite weights in %s were set to 0\n" % k)
        netspec.save_sparse(self.state, os.path.join(self.work, "working.npz"))
        with open(os.path.join(self.work, "passes.json"), "w") as fh:
            json.dump({"used": self.used}, fh)
        text = buf.getvalue()
        if len(text) > OUTPUT_CAP:
            text = (text[:OUTPUT_CAP * 2 // 3] + "\n[... %d characters cut ...]\n"
                    % (len(text) - OUTPUT_CAP) + text[-OUTPUT_CAP // 3:])
        spent, self.log = self.log, []
        return {"output": text, "passes_used": self.used, "passes_spent_on": spent,
                "seconds": round(time.time() - t0, 1), "refused": refused}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    channel = sys.stdout                      # the protocol channel; user prints are redirected
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "solution"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "evaluation"))
    kernel = Kernel(args.base, args.budget, args.work, args.split)
    channel.write(json.dumps({"ready": True, "passes_used": kernel.used}) + "\n")
    channel.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reply = kernel.run(json.loads(line)["code"])
        channel.write(json.dumps(reply) + "\n")
        channel.flush()


if __name__ == "__main__":
    main()
