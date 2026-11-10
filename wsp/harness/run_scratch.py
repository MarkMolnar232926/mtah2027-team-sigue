"""Experiments A and B: a network written from scratch by a tool-free LLM.

    python wsp/harness/run_scratch.py --run expA_scratch_oneshot --passes 0
    python wsp/harness/run_scratch.py --run expB_scratch_passes10 --passes 10

--passes 0   one reply, no feedback (the paper's Exp. 5).
--passes N   N practice passes: after each reply the network is scored on the dev split and the
             next message reports the result; then one final reply (the paper's Exp. 6 idea,
             without a code sandbox).

Per turn, in wsp/runs/<run>/turn_NN/: what was sent, the reply, the event stream, run.json,
apply.json and, for a practice pass, dev_eval.json. Networks go to wsp/networks/<run>/ (pass_NN.npz
= the network scored in pass NN, hand.pt = the final one, loadable like any models/*.pt). The final
network is scored once on the held-out benchmark -> wsp/eval/<run>_benchmark.json.

A run can be interrupted and started again with the same command: finished turns are not re-sent.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import build_prompt  # noqa: E402
import edits  # noqa: E402
import llm  # noqa: E402
import netspec  # noqa: E402


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--passes", type=int, default=0)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--model", default=llm.MODEL)
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(ROOT, "wsp", "runs", args.run)
    net_dir = os.path.join(ROOT, "wsp", "networks", args.run)
    session_file = os.path.join(llm.SESSIONS, args.run, "session.txt")
    state_path = os.path.join(run_dir, "state.json")
    state = read_json(state_path, {"run": args.run, "passes": args.passes, "model": args.model,
                                   "effort": args.effort, "turns_done": 0, "history": [],
                                   "complete": False})
    if state["passes"] != args.passes:
        raise SystemExit("run %s was started with --passes %d" % (args.run, state["passes"]))

    def pass_path(n):
        return os.path.join(net_dir, "pass_%02d.npz" % n)

    def restore(n):
        return netspec.load_sparse(pass_path(n)) if os.path.exists(pass_path(n)) else None

    done = state["turns_done"]
    current = os.path.join(net_dir, "current.npz")
    hand = os.path.join(net_dir, "hand.pt")
    if state["complete"]:
        net = netspec.from_checkpoint(hand)           # only the benchmark is left to do
    elif done and os.path.exists(current):
        net = netspec.load_sparse(current)
    else:
        net = netspec.zero_state()
    sid = open(session_file).read().strip() if done and os.path.exists(session_file) else None
    if state["complete"]:
        message = None
    elif done:
        with open(os.path.join(run_dir, "next_message.txt"), encoding="utf-8") as fh:
            message = fh.read()
    else:
        message = build_prompt.build(args.passes)

    for turn in range(done, args.passes + 1):
        turn_dir = os.path.join(run_dir, "turn_%02d" % turn)
        rec = read_json(os.path.join(turn_dir, "run.json"))
        if rec and rec.get("exit_code") == 0 and not rec.get("is_error") \
                and os.path.exists(os.path.join(turn_dir, "answer.md")):
            with open(os.path.join(turn_dir, "answer.md"), encoding="utf-8") as fh:
                answer = fh.read()                    # the reply arrived before an interruption
        else:
            print("turn %d: asking the model (effort %s) ..." % (turn, args.effort), flush=True)
            answer, sid, rec = llm.run_turn(turn_dir, message, args.run, resume=sid,
                                            effort=args.effort, model=args.model, first=turn == 0)
            os.makedirs(os.path.dirname(session_file), exist_ok=True)
            with open(session_file, "w") as fh:
                fh.write(sid or "")
        print("turn %d: reply %d chars, %s output tokens, %.0f s, tool_free=%s, truncated=%s" % (
            turn, len(answer), rec["usage"]["output_tokens"], rec["wall_seconds"],
            rec["audit"]["tool_free"], rec["truncated"]), flush=True)

        report = edits.apply_text(net, answer, reset=netspec.zero_state, restore=restore)
        write_json(os.path.join(turn_dir, "apply.json"), report)
        print("turn %d: %d statements, %d applied, %d errors, %d nonzero weights" % (
            turn, report["statements"], report["applied"], len(report["errors"]),
            report["nonzero_weights"]), flush=True)
        netspec.save_sparse(net, current)

        if turn < args.passes:
            import evaluate
            import feedback
            pass_no = turn + 1
            netspec.save_sparse(net, pass_path(pass_no))
            result, kept = evaluate.score_state(net, "dev", keep=True)
            write_json(os.path.join(turn_dir, "dev_eval.json"), result)
            all_ = result["sources"]["ALL"]
            state["history"].append([pass_no, all_["spell_f1"], all_["macro_f1"]])
            print("pass %d: dev %s" % (pass_no, evaluate.brief(result)), flush=True)
            message = feedback.message(pass_no, args.passes, report, result, kept,
                                       [tuple(h) for h in state["history"]])
            with open(os.path.join(run_dir, "next_message.txt"), "w", encoding="utf-8",
                      newline="\n") as fh:
                fh.write(message)
        state["turns_done"] = turn + 1
        write_json(state_path, state)

    if not state["complete"]:
        netspec.save_checkpoint(net, hand)
        for leftover in (current, os.path.join(run_dir, "next_message.txt")):
            if os.path.exists(leftover):
                os.remove(leftover)
        state["complete"] = True
        state["nonzero_weights"] = netspec.n_nonzero(net)
        write_json(state_path, state)
    print("final network: wsp/networks/%s/hand.pt (%d nonzero weights of %d)" % (
        args.run, state["nonzero_weights"], netspec.N_PARAMS), flush=True)

    if not args.skip_benchmark:
        import evaluate
        result = evaluate.score_state(net, "benchmark")
        result["network"] = "wsp/networks/%s/hand.pt" % args.run
        write_json(os.path.join(ROOT, "wsp", "eval", "%s_benchmark.json" % args.run), result)
        print("BENCHMARK spellF1 / macroF1   " + evaluate.brief(result), flush=True)


if __name__ == "__main__":
    main()
