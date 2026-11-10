These visible-ball misses are genuinely ambiguous in the inputs (long clearances with both teams 20–40 m from the ball, or ball 10 m from everybody), and the network responds with near-zero margins rather than confident errors. So no single "bug" there. Next measurement: how the spell-level errors are distributed (missed vs spurious, by length), and how much accumulated log-prob evidence missed/spurious spells carry relative to the decoder's 16/24 penalties — this tells me whether the logit scale or the D-vs-live balance is mis-set relative to the decoder.

```python
def spells(lab):
    return [(s, e, l) for s, e, l in runs(lab) if l < 2]

def match(gt, pr, thr=0.5):
    # greedy one-to-one matching by IoU, same label required
    cand = []
    for i, (s, e, l) in enumerate(gt):
        for j, (s2, e2, l2) in enumerate(pr):
            if l2 != l or e2 <= s or s2 >= e: continue
            inter = min(e, e2) - max(s, s2); uni = max(e, e2) - min(s, s2)
            cand.append((inter / uni, i, j))
    cand.sort(reverse=True)
    mg, mp = {}, {}
    for iou, i, j in cand:
        if iou < thr: break
        if i in mg or j in mp: continue
        mg[i] = j; mp[j] = i
    return mg, mp

bins = [(0, 2), (2, 5), (5, 10), (10, 1e9)]
def report(name, pred, verbose=True):
    y = D[name]["y"]
    gt = spells(y); pr = spells(pred)
    mg, mp = match(gt, pr)
    g = [(lo, sum(1 for i, r in enumerate(gt) if lo <= (r[1] - r[0]) / 25 < hi), sum(1 for i, r in enumerate(gt) if lo <= (r[1] - r[0]) / 25 < hi and i in mg)) for lo, hi in bins]
    p = [(lo, sum(1 for j, r in enumerate(pr) if lo <= (r[1] - r[0]) / 25 < hi), sum(1 for j, r in enumerate(pr) if lo <= (r[1] - r[0]) / 25 < hi and j in mp)) for lo, hi in bins]
    if verbose:
        print("  %s GT (len>=, n, matched@.5): %s | PRED: %s" % (name, g, p))
    return g, p

for name in D:
    report(name, D[name]["pred"])

# Why are GT spells unmatched at IoU .5 ?  classify each unmatched GT spell on DFL
def why(name):
    d = D[name]; y = d["y"]; pred = d["pred"]
    gt = spells(y); pr = spells(pred); mg, mp = match(gt, pr)
    cnt = collections.Counter()
    for i, (s, e, l) in enumerate(gt):
        if i in mg: continue
        seg = pred[s:e]; fr = [(seg == k).mean() for k in range(3)]
        if fr[l] >= 0.5:
            # mostly right label but poor IoU => predicted spell extends beyond (merge/over-extension) 
            cnt["right-team, pred spell too long (merged/extended)"] += 1
        elif fr[2] >= 0.5:
            cnt["called dead"] += 1
        else:
            cnt["called other team"] += 1
    return cnt
import collections
for name in ["dfl0", "dfl2", "sc8"]:
    print(name, dict(why(name)))
```