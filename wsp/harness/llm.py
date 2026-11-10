"""One turn = one headless Claude Code call with every tool switched off.

The counterpart of the paper's `codex exec` / `codex exec resume`: the prompt goes in on stdin, the
reply comes back as an event stream, and nothing else can happen in between. The call runs from an
empty folder outside the repository, with a one-line system prompt (instructions.md), no tools, no
MCP servers, no skills, no CLAUDE.md -- so the writer knows only what the prompt says.

Files written into the turn folder:
    prompt.txt | message.txt   what was sent (first turn | later turns)
    events.jsonl               the event stream, scrubbed of local paths and ids
    answer.md                  the reply text
    reasoning.md               the model's reasoning summaries, when the stream carries any
    run.json                   model, effort, command line, tokens, wall time, tool audit
    stderr.log                 the CLI's stderr

    python wsp/harness/llm.py --dir wsp/runs/x/reply --prompt-file p.txt --session-dir x
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-fable-5-2"
SESSIONS = os.path.join(tempfile.gettempdir(), "wsp_sessions")
# Keys of the stream that describe this machine or account rather than the conversation.
DROP_KEYS = {"cwd", "original_cwd", "messaging_socket_path", "powershell_path", "uuid",
             "credential_expires_at", "additional_directories", "apiKeySource", "plugins",
             "rate_limit_info", "parent_tool_use_id"}
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


def system_line():
    with open(os.path.join(HERE, "instructions.md"), encoding="utf-8") as fh:
        return fh.read().strip()


def _scrub(obj, ids):
    """Drop machine/account fields; replace session and message uuids by stable short tokens."""
    if isinstance(obj, dict):
        out = {k: _scrub(v, ids) for k, v in obj.items() if k not in DROP_KEYS}
        if isinstance(obj.get("signature"), str):
            # The model's reasoning is not exposed as text; the stream carries only an opaque
            # signed blob of it (hundreds of KB). Its length is kept, the blob is not.
            out["signature"] = "<opaque reasoning signature, %d characters, omitted>" % len(obj["signature"])
        return out
    if isinstance(obj, list):
        return [_scrub(v, ids) for v in obj]
    if isinstance(obj, str):
        home = os.path.expanduser("~")
        for raw in (home, home.replace("\\", "\\\\"), home.replace("\\", "/")):
            obj = obj.replace(raw, "<home>")
        return UUID.sub(lambda m: ids.setdefault(m.group(0), "id-%08x" % len(ids)), obj)
    return obj


def run_turn(turn_dir, prompt, session_dir, resume=None, effort="xhigh", model=MODEL,
             first=True, timeout=3 * 3600):
    """Send `prompt`, save everything, return (answer text, session id, run record)."""
    os.makedirs(turn_dir, exist_ok=True)
    cwd = os.path.join(SESSIONS, session_dir)
    os.makedirs(cwd, exist_ok=True)
    with open(os.path.join(turn_dir, "prompt.txt" if first else "message.txt"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(prompt)

    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError("the `claude` CLI is not on PATH")
    args = ["-p", "--model", model, "--effort", effort, "--tools", "", "--strict-mcp-config",
            "--safe-mode", "--disable-slash-commands", "--system-prompt", system_line(),
            "--output-format", "stream-json", "--verbose"]
    if resume:
        args += ["--resume", resume]
    env = dict(os.environ)
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "128000"      # a network is a long reply

    t0 = time.time()
    proc = subprocess.run([exe] + args, input=prompt.encode("utf-8"), cwd=cwd, env=env,
                          capture_output=True, timeout=timeout)
    wall = time.time() - t0

    events, texts, thoughts, tool_uses, init, result = [], [], [], [], None, None
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        events.append(e)
        if e.get("type") == "system" and e.get("subtype") == "init":
            init = e
        elif e.get("type") == "assistant":
            for block in e["message"].get("content", []):
                if block.get("type") == "text":
                    texts.append(block["text"])
                elif block.get("type") == "thinking" and block.get("thinking"):
                    thoughts.append(block["thinking"])
                elif block.get("type") in ("tool_use", "server_tool_use"):
                    tool_uses.append(block.get("name"))
        elif e.get("type") == "result":
            result = e

    answer = (result or {}).get("result") or "\n\n".join(texts)
    session_id = (result or init or {}).get("session_id")
    ids = {}
    with open(os.path.join(turn_dir, "events.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for e in events:
            fh.write(json.dumps(_scrub(e, ids), ensure_ascii=False) + "\n")
    with open(os.path.join(turn_dir, "answer.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(answer)
    if thoughts:
        with open(os.path.join(turn_dir, "reasoning.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n\n---\n\n".join(thoughts))
    with open(os.path.join(turn_dir, "stderr.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_scrub(proc.stderr.decode("utf-8", "replace"), ids))

    usage = (result or {}).get("usage") or {}
    tools_listed = (init or {}).get("tools")
    mcp_listed = (init or {}).get("mcp_servers")
    record = {
        "model": (init or {}).get("model", model),
        "effort": effort,
        "cli": "claude " + " ".join(
            '""' if a == "" else "<instructions.md>" if a == system_line()
            else "<session>" if a == resume else a for a in args),
        "cli_version": (init or {}).get("claude_code_version"),
        "system_prompt": system_line(),
        "resumed": bool(resume),
        "exit_code": proc.returncode,
        "wall_seconds": round(wall, 1),
        "stop_reason": (result or {}).get("stop_reason"),
        "is_error": (result or {}).get("is_error"),
        "num_turns": (result or {}).get("num_turns"),
        "usage": {
            "input_tokens": usage.get("input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        },
        "reasoning_text_exposed": bool(thoughts),
        "prompt_chars": len(prompt),
        "answer_chars": len(answer),
        "audit": {
            # tool-free = the session had no tool to call and the reply contains no tool call
            "tools_available": tools_listed,
            "mcp_servers": mcp_listed,
            "tool_calls_in_reply": tool_uses,
            "tool_free": tools_listed == [] and mcp_listed == [] and not tool_uses,
        },
        "truncated": (result or {}).get("stop_reason") == "max_tokens",
    }
    with open(os.path.join(turn_dir, "run.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=2)
    if proc.returncode != 0 or result is None or record["is_error"]:
        raise RuntimeError("claude call failed (exit %s); see %s" % (proc.returncode, turn_dir))
    return answer, session_id, record


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="turn folder to write into")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--session-dir", required=True, help="name of the empty working folder")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    with open(a.prompt_file, encoding="utf-8") as fh:
        text = fh.read()
    _, sid, rec = run_turn(a.dir, text, a.session_dir, resume=a.resume, effort=a.effort,
                           model=a.model, first=a.resume is None)
    print("session", sid)
    print(json.dumps(rec["usage"]), "tool_free", rec["audit"]["tool_free"])
