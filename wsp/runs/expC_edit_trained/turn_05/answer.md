Both layer-2 GRU directions carry the team decision redundantly; the direct `h` path into the head is nearly irrelevant for team (0.60–0.67 team accuracy alone, removing it changes little). So the decision is made in the recurrent stack. Let me look at raw traces across three visible-ball missed possessions to see what the inputs say versus what the network holds.

```python
def trace(name, t0, t1, step=0.4):
    d = D[name]; x = d["x"]; y = d["y"]; lg = d["logits"]
    lgf = head(d, ablate=[1, 2]); lgb = head(d, ablate=[0, 2])
    print("%s  t   y p | H-A  (qf  qb) D-liv | seen d1H  d1A  relvH relvA n2H n2A spd  since" % name)
    for t in np.arange(t0, t1, step):
        i = int(round(t * 25)); j = int(round((t + step) * 25))
        sl = slice(i, j)
        xm = x[sl].mean(0)
        print("%6.1f  %s %s | %+5.1f (%+5.1f %+5.1f) %+5.1f | %.1f  %4.1f %4.1f  %4.1f %4.1f  %.1f %.1f  %4.1f %4.1f" % (
            t, L[y[i]], L[d["pred"][i]], (lg[sl, 0] - lg[sl, 1]).mean(), (lgf[sl, 0] - lgf[sl, 1]).mean(), (lgb[sl, 0] - lgb[sl, 1]).mean(),
            (lg[sl, 2] - lg[sl, :2].max(1)).mean(), xm[0], xm[13] * 10, xm[30] * 10, xm[18] * 10, xm[35] * 10, xm[15], xm[32], xm[3] * 10, xm[5] * 10))

trace("dfl2", 243.0, 251.5)
print()
trace("dfl2", 637.5, 647.0)
print()
trace("dfl2", 205.5, 214.5)
```