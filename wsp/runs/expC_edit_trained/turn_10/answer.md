Turnover boundaries on DFL are noisy (only ~half of the possessions matched at IoU 0.3 survive at 0.7), with a mild tendency to end late. To study this causally I'll build an offline simulator of the full network from stored inputs (free), validate it against the stored logits, and emulate the four-view pipeline by transforming `x` (y-mirror: negate y features; role swap: exchange team blocks, negate ball x/y, centroids and d1_diff, swap H/A outputs).

```python
def sig(v): return 1.0 / (1.0 + np.exp(-v))

def run_gru(X, Wd, pre, reverse=False):
    # X: (B,T,I) -> (B,T,128)
    Wir, Wiz, Win = Wd[pre + ".Wir"], Wd[pre + ".Wiz"], Wd[pre + ".Win"]
    Whr, Whz, Whn = Wd[pre + ".Whr"], Wd[pre + ".Whz"], Wd[pre + ".Whn"]
    B, T, I = X.shape
    Gr = X @ Wir.T + (Wd[pre + ".bir"] + Wd[pre + ".bhr"])
    Gz = X @ Wiz.T + (Wd[pre + ".biz"] + Wd[pre + ".bhz"])
    Gn = X @ Win.T + Wd[pre + ".bin"]
    bhn = Wd[pre + ".bhn"]
    WhrT, WhzT, WhnT = Whr.T.copy(), Whz.T.copy(), Whn.T.copy()
    s = np.zeros((B, 128), np.float32)
    out = np.empty((B, T, 128), np.float32)
    rng = range(T - 1, -1, -1) if reverse else range(T)
    for t in rng:
        r = sig(Gr[:, t] + s @ WhrT)
        z = sig(Gz[:, t] + s @ WhzT)
        n = np.tanh(Gn[:, t] + r * (s @ WhnT + bhn))
        s = (1 - z) * n + z * s
        out[:, t] = s
    return out

def front_end(X, Wd):
    # X (B,T,48) -> h (B,T,128)
    a = gelu(X @ Wd["inp.W"].T + Wd["inp.b"]).astype(np.float32)
    B, T, C = a.shape
    ap = np.concatenate([np.zeros((B, 4, C), np.float32), a, np.zeros((B, 4, C), np.float32)], 1)
    c = np.zeros((B, T, 128), np.float32) + Wd["conv.b"]
    cw = Wd["conv.W"]
    for k in range(9):
        c += ap[:, k:k + T] @ cw[:, :, k].T
    return a + np.maximum(c, 0)

def back_end(h, Wd):
    g1 = np.concatenate([run_gru(h, Wd, "gru1f"), run_gru(h, Wd, "gru1b", True)], -1)
    qf = run_gru(g1, Wd, "gru2f"); qb = run_gru(g1, Wd, "gru2b", True)
    pre = qf @ Wd["head1.Wf"].T + qb @ Wd["head1.Wb"].T + h @ Wd["head1.Wh"].T + Wd["head1.b"]
    m = gelu(pre)
    return m @ Wd["head2.W"].T + Wd["head2.b"], g1, np.concatenate([qf, qb], -1)

HB, AB = 13, 30
def views(x):
    # returns (4,T,48): orig, mirror-y, swap, swap+mirror
    def mir(v):
        v = v.copy()
        for i in (2, 8, HB + 9, AB + 9): v[:, i] = -v[:, i]
        return v
    def swp(v):
        o = v.copy()
        o[:, HB:HB + 17] = v[:, AB:AB + 17]; o[:, AB:AB + 17] = v[:, HB:HB + 17]
        for i in (1, 2, 7, 8, HB + 8, HB + 9, AB + 8, AB + 9, 47): o[:, i] = -o[:, i]
        return o
    return np.stack([x, mir(x), swp(x), mir(swp(x))], 0)

def sim(name, Wd=None, nviews=4):
    Wd = Wd or W
    X = views(D[name]["x"])[:nviews]
    h = front_end(X, Wd)
    lg, g1, g2 = back_end(h, Wd)
    return lg, h, g1, g2

def pipeline(lg):
    # lg (4,T,3) -> averaged log-prob (T,3)
    p = np.exp(np.stack([lsm(l.astype(np.float64)) for l in lg], 0))
    if p.shape[0] == 4:
        p[2:] = p[2:][:, :, [1, 0, 2]]
    pm = p.mean(0)
    return np.log(np.clip(pm, 1e-6, 1.0))

import time
t0 = time.time()
S = {}
for name in D:
    lg, h, g1, g2 = sim(name)
    S[name] = dict(lg=lg)
    d = D[name]
    sl = slice(1000, 23000)
    print(name, "sim vs stored (orig view, interior): h maxdiff %.4f  logits maxdiff %.4f  mean abs %.5f" % (
        np.abs(h[0, sl] - d["h"][sl]).max(), np.abs(lg[0, sl] - d["logits"][sl]).max(), np.abs(lg[0, sl] - d["logits"][sl]).mean()))
    y = d["y"][sl]
    p1 = decode(np.maximum(lsm(lg[0, sl].astype(np.float64)), math.log(1e-6)))
    p4 = decode(pipeline(lg[:, sl]))
    S[name]["pred4"] = p4
    print("    single view:", metrics(y, p1), " acc %.4f" % (p1 == y).mean())
    print("    four views :", metrics(y, p4), " acc %.4f" % (p4 == y).mean())
    # view agreement on team among live frames
    live = y < 2
    tm = np.stack([(lg[v, sl][live][:, 1] > lg[v, sl][live][:, 0]) for v in range(4)], 0)
    tm[2:] = ~tm[2:]
    acc = [(tm[v] == (y[live] == 1)).mean() for v in range(4)]
    print("    per-view team acc", np.round(acc, 3), " unanimous frac %.3f" % ((tm.sum(0) == 0) | (tm.sum(0) == 4)).mean())
print("time", time.time() - t0)
```