Now a batched evaluation harness over all 13 chunks × 4 views, and baseline diagnostics: per-source spellF1 for the four-view pipeline, for the original-orientation views only, and for the swapped views only (held-out proxy).

```python
SL = slice(1000, 23000)
def full_logits(Wd=None, names=NAMES, groups=None, hfun=None):
    Wd = Wd or W
    out = {}
    groups = groups or [g for g in GROUPS]
    for g in groups:
        g = [n for n in g if n in names]
        if not g: continue
        X = np.concatenate([views(D[n]["x"]) for n in g], 0)
        h = front_end(X, Wd)
        if hfun is not None: h = hfun(h)
        lg, g1, g2 = back_end(h, Wd)
        for i, n in enumerate(g):
            out[n] = lg[4 * i:4 * i + 4].astype(np.float32)
    return out

def pipe_views(lg, vs):
    p = np.exp(np.stack([lsm(lg[v].astype(np.float64)) for v in vs], 0))
    for i, v in enumerate(vs):
        if v >= 2: p[i] = p[i][:, [1, 0, 2]]
    return np.log(np.clip(p.mean(0), 1e-6, 1.0))

def summ(LG, tag="", vsets=((0, 1, 2, 3),), verbose=True, keep=None):
    res = {}
    for vs in vsets:
        per = collections.defaultdict(list)
        for n, lg in LG.items():
            y = D[n]["y"][SL]
            pr = decode(pipe_views(lg[:, SL], vs))
            if keep is not None: keep[(n, vs)] = pr
            mt = metrics(y, pr)
            live = y < 2
            src = "DFL" if n.startswith("dfl") else ("SC" if n.startswith("sc") else "MET")
            per[src].append((mt["spell_f1"], mt["macro_f1"], mt["n_pred_spells"], mt["n_gt_spells"], (pr == y).mean()))
        s = "%-22s views %-9s" % (tag, "".join(map(str, vs)))
        for src in ("DFL", "SC", "MET"):
            if src in per:
                a = np.array(per[src])
                s += " | %s sF1 %.4f mF1 %.4f n %3d/%3d acc %.4f" % (src, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].sum(), a[:, 3].sum(), a[:, 4].mean())
                res[(src, vs)] = a
        if verbose: print(s)
    return res

LG0 = full_logits(BASE)
PRED0 = {}
R0 = summ(LG0, "BASE", vsets=((0, 1, 2, 3), (0, 1), (2, 3), (0,)), keep=PRED0)
print("per-chunk 4-view spellF1:", {n: round(float(metrics(D[n]["y"][SL], PRED0[(n, (0, 1, 2, 3))])["spell_f1"]), 3) for n in NAMES})
```