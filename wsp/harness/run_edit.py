"""Experiment C: a trained network improved by editing its weights (the paper's Exp. 4 / Exp. 8).

    python wsp/harness/run_edit.py --run expC_edit_trained --base models/expB_bw3met6k_s0.pt

Each turn the tool-free model writes one Python block; it runs in the guarded analysis kernel
(kernel.py) and the printed output is the next message. The model can read and write every weight
and has a budget of forward passes on the dev split. A reply without a Python block ends the run.

Per turn, in wsp/runs/<run>/turn_NN/: what was sent, the reply, the event stream, run.json. All
code and outputs are in wsp/runs/<run>/tool_calls.jsonl. Results in wsp/networks/<run>/: edited.pt
(loadable like any models/*.pt) and diff.npz (edited - base, changed tensors only). Base and edited
are then scored on the held-out benchmark -> wsp/eval/<run>_benchmark.json.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build_prompt  # noqa: E402
import kernel as kernel_mod  # noqa: E402
import llm  # noqa: E402
import netspec  # noqa: E402
from run_scratch import read_json, write_json  # noqa: E402

PY_BLOCK = re.compile(r"```python[^\n]*\n(.*?)```", re.S)


class KernelProc:
    def __init__(self, base, budget, work, timeout_s):
        self.cmd = [sys.executable, "-u", os.path.join(HERE, "kernel.py"), "--base", base,
                    "--budget", str(budget), "--work", work]
        self.work, self.timeout_s, self.budget = work, timeout_s, budget
        os.makedirs(work, exist_ok=True)
        self.start()

    def start(self):
        self.err = open(os.path.join(self.work, "kernel_stderr.log"), "a")
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, cwd=ROOT, text=True, encoding="utf-8")
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()
        ready = self._read(600)
        if not ready or not ready.get("ready"):
            raise RuntimeError("the analysis kernel did not start; see %s" % self.work)
        self.used = ready["passes_used"]

    @staticmethod
    def _pump(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _read(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        return json.loads(line) if line else None

    def run(self, code):
        self.proc.stdin.write(json.dumps({"code": code}) + "\n")
        self.proc.stdin.flush()
        reply = self._read(self.timeout_s)
        if reply is None:                         # too slow, or the process died: start again
            self.proc.kill()
            self.proc.wait()
            self.start()
            reply = {"output": "[kernel] The block ran longer than %d minutes (or the process "
                               "stopped) and the Python process was restarted. Your weight edits "
                               "up to the previous block are kept; your variables are gone."
                               % (self.timeout_s // 60),
                     "passes_used": self.used, "passes_spent_on": [], "seconds": self.timeout_s,
                     "refused": False}
        self.used = reply["passes_used"]
        return reply

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        self.err.close()


def diff_report(base, edited):
    tensors, total = {}, 0
    delta = {}
    for k in base:
        d = edited[k].astype(np.float64) - base[k].astype(np.float64)
        changed = int(np.count_nonzero(d))
        if changed:
            tensors[k] = {"changed": changed, "size": int(d.size),
                          "max_abs_change": float(np.abs(d).max()),
                          "rms_change": float(np.sqrt((d ** 2).mean())),
                          "rms_base": float(np.sqrt((base[k].astype(np.float64) ** 2).mean()))}
            delta[k] = (edited[k] - base[k]).astype(np.float32)
            total += changed
    return {"weights_changed": total, "weights_total": netspec.N_PARAMS,
            "share_changed": total / netspec.N_PARAMS, "tensors": tensors}, delta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="models/expB_bw3met6k_s0.pt")
    ap.add_argument("--budget", type=int, default=25, help="forward passes")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--block-minutes", type=int, default=20)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--model", default=llm.MODEL)
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(ROOT, "wsp", "runs", args.run)
    net_dir = os.path.join(ROOT, "wsp", "networks", args.run)
    work = os.path.join(llm.SESSIONS, args.run, "kernel")
    session_file = os.path.join(llm.SESSIONS, args.run, "session.txt")
    state_path = os.path.join(run_dir, "state.json")
    state = read_json(state_path, {"run": args.run, "base": args.base, "budget": args.budget,
                                   "max_turns": args.max_turns, "model": args.model,
                                   "effort": args.effort, "turns_done": 0, "passes_used": 0,
                                   "complete": False})
    base = netspec.from_checkpoint(os.path.join(ROOT, args.base))
    edited_path = os.path.join(net_dir, "edited.pt")

    if not state["complete"]:
        kern = KernelProc(os.path.join(ROOT, args.base), args.budget, work, args.block_minutes * 60)
        done = state["turns_done"]
        sid = open(session_file).read().strip() if done and os.path.exists(session_file) else None
        if done:
            with open(os.path.join(run_dir, "next_message.txt"), encoding="utf-8") as fh:
                message = fh.read()
        else:
            message = build_prompt.build_edit(args.budget, args.max_turns, args.block_minutes,
                                              kernel_mod.OUTPUT_CAP, kernel_mod.API_DOC)
        final_answer = None
        for turn in range(done, args.max_turns):
            turn_dir = os.path.join(run_dir, "turn_%02d" % turn)
            rec = read_json(os.path.join(turn_dir, "run.json"))
            if rec and rec.get("exit_code") == 0 and not rec.get("is_error") \
                    and os.path.exists(os.path.join(turn_dir, "answer.md")):
                with open(os.path.join(turn_dir, "answer.md"), encoding="utf-8") as fh:
                    answer = fh.read()
            else:
                print("turn %d: asking the model (effort %s) ..." % (turn, args.effort), flush=True)
                answer, sid, rec = llm.run_turn(turn_dir, message, args.run, resume=sid,
                                                effort=args.effort, model=args.model,
                                                first=turn == 0)
                os.makedirs(os.path.dirname(session_file), exist_ok=True)
                with open(session_file, "w") as fh:
                    fh.write(sid or "")
            blocks = PY_BLOCK.findall(answer)
            print("turn %d: reply %d chars, %s output tokens, %.0f s, tool_free=%s, code=%s" % (
                turn, len(answer), rec["usage"]["output_tokens"], rec["wall_seconds"],
                rec["audit"]["tool_free"], bool(blocks)), flush=True)
            if not blocks:
                final_answer = answer
                state["turns_done"] = turn + 1
                write_json(state_path, state)
                break
            out = kern.run(blocks[-1])
            with open(os.path.join(run_dir, "tool_calls.jsonl"), "a", encoding="utf-8",
                      newline="\n") as fh:
                fh.write(json.dumps({"turn": turn, "code": blocks[-1], "output": out["output"],
                                     "seconds": out["seconds"], "refused": out["refused"],
                                     "passes_spent_on": out["passes_spent_on"],
                                     "passes_used": out["passes_used"]}, ensure_ascii=False) + "\n")
            print("turn %d: block ran %.0f s, passes used %d/%d%s" % (
                turn, out["seconds"], out["passes_used"], args.budget,
                " (REFUSED)" if out["refused"] else ""), flush=True)
            left = args.max_turns - turn - 1
            message = ("OUTPUT of your block (ran %.0f s). Forward passes used: %d of %d. Replies "
                       "left: %d.\n----\n%s\n----\n" % (out["seconds"], out["passes_used"],
                                                        args.budget, left, out["output"].rstrip()))
            if len(blocks) > 1:
                message += "Note: your reply had %d python blocks; only the last one was run.\n" % len(blocks)
            if left == 1:
                message += ("This is your LAST reply. Do not write a python block: give your final "
                            "answer as described under HOW TO REPLY. The working network as it "
                            "stands now is the result.\n")
            elif left <= 3:
                message += "You have %d replies left; the last one must be your final answer.\n" % left
            with open(os.path.join(run_dir, "next_message.txt"), "w", encoding="utf-8",
                      newline="\n") as fh:
                fh.write(message)
            state["turns_done"], state["passes_used"] = turn + 1, out["passes_used"]
            write_json(state_path, state)
        state["passes_used"] = kern.used
        kern.close()

        edited = netspec.load_sparse(os.path.join(work, "working.npz")) \
            if os.path.exists(os.path.join(work, "working.npz")) else netspec.copy_state(base)
        report, delta = diff_report(base, edited)
        netspec.save_checkpoint(edited, edited_path)
        np.savez_compressed(os.path.join(net_dir, "diff.npz"), **delta)
        write_json(os.path.join(run_dir, "edit_report.json"), report)
        if final_answer is not None:
            with open(os.path.join(run_dir, "final_answer.md"), "w", encoding="utf-8",
                      newline="\n") as fh:
                fh.write(final_answer)
        leftover = os.path.join(run_dir, "next_message.txt")
        if os.path.exists(leftover):
            os.remove(leftover)
        state.update(complete=True, finished_with_final_answer=final_answer is not None,
                     weights_changed=report["weights_changed"])
        write_json(state_path, state)
        print("edited network: wsp/networks/%s/edited.pt  (%d of %d weights changed, %d passes used)"
              % (args.run, report["weights_changed"], netspec.N_PARAMS, state["passes_used"]),
              flush=True)

    if not args.skip_benchmark:
        import evaluate
        edited = netspec.from_checkpoint(edited_path)
        res_base = evaluate.score_state(base, "benchmark")
        res_edit = evaluate.score_state(edited, "benchmark")
        write_json(os.path.join(ROOT, "wsp", "eval", "%s_benchmark.json" % args.run),
                   {"base": dict(res_base, network=args.base),
                    "edited": dict(res_edit, network="wsp/networks/%s/edited.pt" % args.run)})
        print("BENCHMARK base    " + evaluate.brief(res_base))
        print("BENCHMARK edited  " + evaluate.brief(res_edit), flush=True)


if __name__ == "__main__":
    main()
