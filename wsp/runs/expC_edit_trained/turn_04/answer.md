Missed spells have small negative margins (−0.5…−2), while correct ones sit at +5…+10; several misses are with the ball fully visible (short turnovers continued as the previous team). Next: (1) which of the three paths into the head (forward GRU, backward GRU, local h) carries the team decision and which is right when the network is wrong, via mean-ablation computed offline; (2) whether simple local evidence (d1_diff) contradicts the network in the missed spells.

```python
def gelu(v):
    from math import sqrt
    # erf via numpy-free approximation: use tanh-free exact formula with np special not available -> series approx
    return v * 0.5 * (1.0 + erf_np(v / np.sqrt(2.0)))

def erf_np(x):
    # Abramowitz-Stegun 7.1.26, max abs err 1.5e-7
    s = np.sign(x); ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    yv = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return s * yv

def head(d, Wd=None, scale=(1, 1, 1), ablate=None, means=None):
    Wd = Wd or W
    qf = d["g2"][:, :128]; qb = d["g2"][:, 128:]; h = d["h"]
    pf = qf @ Wd["head1.Wf"].T; pb = qb @ Wd["head1.Wb"].T; ph = h @ Wd["head1.Wh"].T
    parts = [pf, pb, ph]
    if ablate is not None:
        for i in ablate:
            parts[i] = np.broadcast_to(parts[i].mean(0, keepdims=True), parts[i].shape)
    pre = scale[0] * parts[0] + scale[1] * parts[1] + scale[2] * parts[2] + Wd["head1.b"]
    m = gelu(pre)
    return m @ Wd["head2.W"].T + Wd["head2.b"]

def lsm(lg):
    z = lg - lg.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))

# check reproduction
for name, d in D.items():
    lg = head(d)
    print(name, "max abs diff vs stored logits", float(np.abs(lg - d["logits"]).max()))

print("\nmean-ablation of paths (single view). cols: argmax acc | decoded acc | spellF1 | n_pred | team-acc on live frames (argmax H vs A)")
for name, d in D.items():
    y = d["y"]; live = y < 2
    for lab, abl in [("full", None), ("no qf", [0]), ("no qb", [1]), ("no h", [2]), ("only qf", [1, 2]), ("only qb", [0, 2]), ("only h", [0, 1])]:
        lg = head(d, ablate=abl)
        lp = lsm(lg); pr = decode(lp); mt = metrics(y, pr)
        team = (lg[live, 1] > lg[live, 0]).astype(int)
        print("  %-6s %-8s %.3f | %.3f | %.3f | %3d (gt %d) | %.3f" % (name, lab, (lg.argmax(1) == y).mean(), (pr == y).mean(),
              mt["spell_f1"], mt["n_pred_spells"], mt["n_gt_spells"], (team == y[live]).mean()))
```