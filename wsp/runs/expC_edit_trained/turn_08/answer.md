Logit scale and D offset are both flat around the current value — the readout is already matched to the decoder, so no gain there. Two useful realisations: (1) since I hold `x`, `h`, `g1`, `g2`, I can re-simulate everything from the inputs onward offline for free (and even emulate the mirrored/role-swapped views by transforming `x`); (2) spellF1 is very sensitive to boundary timing. Let me first measure boundary timing bias on matched possessions and how much is lost between IoU 0.3 and 0.7.

```python
def boundary_stats(name, pred=None):
    d = D[name]; y = d["y"]; pred = d["pred"] if pred is None else pred
    gt = spells(y); pr = spells(pred)
    mg, mp = match(gt, pr, thr=0.3)
    n = len(y)
    off = collections.defaultdict(list)
    ious = []
    for i, j in mg.items():
        s, e, l = gt[i]; s2, e2, _ = pr[j]
        ious.append((min(e, e2) - max(s, s2)) / (max(e, e2) - min(s, s2)))
        if s > 0:
            kind = "start after D" if y[s - 1] == 2 else "start after turnover"
            off[kind].append((s2 - s) / 25)
        if e < n:
            kind = "end into D" if y[e] == 2 else "end by turnover"
            off[kind].append((e2 - e) / 25)
    ious = np.array(ious)
    print(name, "matched@0.3: %d of %d gt, %d pred | of those IoU>=.5: %d, >=.7: %d, >=.9: %d" % (len(mg), len(gt), len(pr), (ious >= .5).sum(), (ious >= .7).sum(), (ious >= .9).sum()))
    for k, v in sorted(off.items()):
        v = np.array(v)
        print("    %-22s n=%3d  mean %+5.2fs  median %+5.2fs  |err|>1s: %2d  late>1s: %2d early>1s: %2d   q10 %+5.2f q90 %+5.2f" % (
            k, len(v), v.mean(), np.median(v), (np.abs(v) > 1).sum(), (v > 1).sum(), (v < -1).sum(), np.quantile(v, .1), np.quantile(v, .9)))

for name in D:
    boundary_stats(name)
```