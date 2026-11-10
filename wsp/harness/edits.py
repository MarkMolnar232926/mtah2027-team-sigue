"""The edits language: how a reply sets network weights.

A reply ends with a fenced block opened by ```edits. One statement per line:

    inp.W[3, 47] = -2.5                 one weight
    conv.W[10, 3, :] = 0.111            a scalar fills a range
    head2.W[0, 0:3] = [4, -4, 0.5]      a list, numpy assignment semantics
    gru1f.bhz[0:8] += 2                 in place: =  +=  -=  *=
    reset                               every weight back to its starting value
    restore 3                           back to the network as it stood after pass 3

Indices are integers, a:b, a:b:s or ':'. A long list may run over several lines. '#' starts a
comment. Statements apply in order; only the last edits block of a reply is read; a statement that
cannot be read is skipped and reported, the rest still apply.

Nothing is evaluated as code: the left side is matched by a regular expression and the right side
is read with ast.literal_eval, so only numbers and lists of numbers get through.
"""
import ast
import re

import numpy as np

from netspec import ALIASES, n_nonzero, summary, view

MAX_ABS = 1e6
BLOCK = re.compile(r"```edits[^\n]*\n(.*?)```", re.S)
STATEMENT = re.compile(r"^([A-Za-z_][\w.]*)\s*(?:\[(.*?)\])?\s*(\+=|-=|\*=|=)\s*(.+)$", re.S)
INT = re.compile(r"^[+-]?\d+$")


class EditError(Exception):
    pass


def last_block(text):
    """Body of the last ```edits block, or None."""
    found = BLOCK.findall(text)
    return found[-1] if found else None


def _strip_comment(line):
    return line.split("#", 1)[0].rstrip()


def statements(block):
    """(first line number, text) per statement; a statement continues while brackets are open."""
    out, buf, start, depth = [], [], 0, 0
    for n, raw in enumerate(block.split("\n"), start=1):
        line = _strip_comment(raw)
        if not line.strip() and depth == 0:
            continue
        if depth == 0:
            buf, start = [], n
        buf.append(line.strip())
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            out.append((start, " ".join(buf)))
            depth = 0
    if depth > 0:
        out.append((start, " ".join(buf)))
    return out


def _index(text, shape):
    """'3, 0:8, :' -> a tuple of ints and slices, checked against the tensor's shape."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) > len(shape):
        raise EditError("%d indices for a tensor with %d axes" % (len(parts), len(shape)))
    idx = []
    for part, size in zip(parts, shape):
        if INT.match(part):
            i = int(part)
            if not -size <= i < size:
                raise EditError("index %d out of range for axis of size %d" % (i, size))
            idx.append(i)
        elif ":" in part:
            fields = part.split(":")
            if len(fields) > 3 or any(f.strip() and not INT.match(f.strip()) for f in fields):
                raise EditError("cannot read range %r" % part)
            vals = [int(f) if f.strip() else None for f in fields] + [None] * (3 - len(fields))
            for v in vals[:2]:
                if v is not None and not -size <= v <= size:
                    raise EditError("range %r out of bounds for axis of size %d" % (part, size))
            if vals[2] == 0:
                raise EditError("range step 0 in %r" % part)
            idx.append(slice(*vals))
        else:
            raise EditError("cannot read index %r (integers and a:b:s ranges only)" % part)
    return tuple(idx)


def _value(text):
    try:
        val = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        raise EditError("right-hand side is not a number or a list of numbers")
    try:
        arr = np.asarray(val, dtype=np.float64)
    except (ValueError, TypeError):
        raise EditError("right-hand side is not a rectangular list of numbers")
    if arr.dtype == object or not np.all(np.isfinite(arr)):
        raise EditError("right-hand side has a non-finite or non-numeric entry")
    if arr.size and np.abs(arr).max() > MAX_ABS:
        raise EditError("value larger than %g" % MAX_ABS)
    return arr


def apply_statement(state, text):
    m = STATEMENT.match(text)
    if not m:
        raise EditError("not of the form NAME[index] = value")
    name, index, op, rhs = m.groups()
    if name not in ALIASES:
        raise EditError("unknown tensor %r" % name)
    target = view(state, name)
    idx = _index(index, target.shape) if index is not None and index.strip() else ()
    val = _value(rhs)
    try:
        if op == "=":
            target[idx] = val
        elif op == "+=":
            target[idx] += val
        elif op == "-=":
            target[idx] -= val
        else:
            target[idx] *= val
    except (ValueError, IndexError) as exc:
        raise EditError("shape mismatch: %s" % exc)
    if not np.all(np.isfinite(target)) or np.abs(target).max() > MAX_ABS:
        raise EditError("result is non-finite or larger than %g" % MAX_ABS)


def apply_text(state, text, reset=None, restore=None):
    """Apply the last edits block of `text` to `state` in place. Returns the report for apply.json.

    reset()      -> a fresh state to start again from (`reset` statement)
    restore(n)   -> the state after pass n               (`restore n` statement)
    Either may be None, in which case the statement is reported as an error.
    """
    block = last_block(text)
    report = {"found_block": block is not None, "statements": 0, "applied": 0, "errors": []}
    if block is None:
        report["errors"].append({"line": 0, "text": "", "reason": "no ```edits block in the reply"})
    else:
        for line, stmt in statements(block):
            report["statements"] += 1
            snapshot = None
            try:
                word = stmt.split()
                if word[0] == "reset" and len(word) == 1:
                    if reset is None:
                        raise EditError("reset is not available in this run")
                    new = reset()
                elif word[0] == "restore" and len(word) == 2 and INT.match(word[1]):
                    if restore is None:
                        raise EditError("restore is not available in this run")
                    new = restore(int(word[1]))
                    if new is None:
                        raise EditError("there is no pass %s to restore" % word[1])
                else:
                    new = None
                    m = STATEMENT.match(stmt)
                    if m and m.group(1) in ALIASES:           # keep a failed in-place op harmless
                        key = ALIASES[m.group(1)][0]
                        snapshot = (key, state[key].copy())
                    apply_statement(state, stmt)
                if new is not None:
                    for k in state:
                        state[k][...] = new[k]
                report["applied"] += 1
            except EditError as exc:
                if snapshot is not None:
                    state[snapshot[0]][...] = snapshot[1]
                report["errors"].append({"line": line, "text": stmt[:200], "reason": str(exc)})
    report["nonzero_weights"] = n_nonzero(state)
    report["tensors"] = {k: v for k, v in summary(state).items() if v["nonzero"]}
    return report
