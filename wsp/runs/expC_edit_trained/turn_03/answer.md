The logits are very sharp (median |H−A| gap 5–12), so decode ≈ argmax: the decoder is hardly doing anything, the network itself fixes the spell structure. On DFL it is over-confident (mean max prob 0.95 vs 0.89 accuracy). Let me look at the anatomy of the errors on DFL: for each true run, what was predicted, the mean logit margins, and ball visibility.

```python
L = "HAD"
def timeline(name, maxlines=95):
    d = D[name]; y = d["y"]; pred = d["pred"]; lg = d["logits"]; x = d["x"]
    lines = []
    for s, e, lab in runs(y):
        seg = pred[s:e]
        fr = [(seg == k).mean() for k in range(3)]
        okfrac = fr[lab]
        # margin of true label vs best other, mean
        other = np.max(np.delete(lg[s:e], lab, axis=1), axis=1)
        marg = lg[s:e, lab] - other
        flag = "" if okfrac > 0.8 else ("  <<< MISSED" if okfrac < 0.5 else "  < partial")
        lines.append("%6.1fs %s %5.1fs pred H%.2f A%.2f D%.2f  margin mean %+5.1f min %+5.1f  seen %.2f%s" % (
            s / 25, L[lab], (e - s) / 25, fr[0], fr[1], fr[2], marg.mean(), marg.min(), x[s:e, 0].mean(), flag))
    return lines

ls = timeline("dfl2")
bad = [l for l in ls if "<" in l]
print(len(ls), "gt runs; flagged", len(bad))
print("\n".join(ls[:90]))
```