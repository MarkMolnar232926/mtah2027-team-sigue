The network has largely memorised MET and SC practice halves (0.98 / 0.90 vs ~0.72–0.84 held-out), so DFL (0.727, 497 predicted vs 596 true possessions) is the most representative of held-out behaviour. I'll collect activations on four chunks (2 DFL, 1 SC, 1 MET) and characterise the errors in the single-view decode.

```python
D = {}
def grab(name, half, start, n=24000):
    o = forward(half, start=start, n=n)
    D[name] = {k: o[k] for k in ["x", "y", "h", "g1", "g2", "m", "logits", "logp"]}

grab("dfl0", 0, 20000)
grab("dfl2", 2, 30000)
grab("sc8", 8, 20000)
grab("met10", 10, 20000)

def runs(lab):
    # list of (start, end_exclusive, label)
    ch = np.flatnonzero(np.diff(lab)) + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(lab)]])
    return [(int(s), int(e), int(lab[s])) for s, e in zip(st, en)]

for name, d in D.items():
    y = d["y"]; lp = d["logp"]
    pred = decode(lp)
    d["pred"] = pred
    am = lp.argmax(1)
    mt = metrics(y, pred)
    print(name, mt)
    print("   argmax acc", float((am == y).mean()), "decoded acc", float((pred == y).mean()))
    conf = np.zeros((3, 3), int)
    for i in range(3):
        for j in range(3):
            conf[i, j] = int(((y == i) & (pred == j)).sum())
    print("   conf", conf.tolist())
    # GT spell length distribution & how many are 'missed' (majority of frames predicted otherwise)
    rs = [r for r in runs(y) if r[2] < 2]
    bins = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]
    out = []
    for lo, hi in bins:
        sel = [r for r in rs if lo <= (r[1] - r[0]) / 25 < hi]
        miss = [r for r in sel if (pred[r[0]:r[1]] == r[2]).mean() < 0.5]
        out.append((lo, hi, len(sel), len(miss)))
    print("   gt spells by length (lo,hi,n,missed):", out)
    # mean confidence
    p = np.exp(lp)
    print("   mean max prob", float(p.max(1).mean()), " mean logp of true", float(lp[np.arange(len(y)), y].mean()),
          " median |H-A| logit gap on live frames", float(np.median(np.abs(d["logits"][y < 2, 0] - d["logits"][y < 2, 1]))))
print(passes_left())
```