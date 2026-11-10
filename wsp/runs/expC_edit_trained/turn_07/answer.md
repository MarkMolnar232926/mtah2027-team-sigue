Unmatched true possessions on DFL split roughly evenly between "called the other team" and "right team but merged into an over-long predicted spell" (the knock-on of the former), with a few "called dead". Before going deeper into the recurrent stack, I want to measure how sensitive the decoded result is to the two quantities that set the logits' relation to the decoder penalties: overall logit scale and the D-vs-live offset. This is a sensitivity measurement on stored logits (free), not something I'll adopt without a mechanism.

```python
def evalset(fn, names=("dfl0", "dfl2", "sc8", "met10")):
    out = []
    for n in names:
        d = D[n]
        lg = fn(d["logits"].astype(np.float64))
        lp = np.maximum(lsm(lg), math.log(1e-6))
        pr = decode(lp); mt = metrics(d["y"], pr)
        out.append((mt["spell_f1"], mt["n_pred_spells"], mt["macro_f1"]))
    return out

def show(tag, res):
    print("%-14s" % tag + " | ".join("%.3f n%2d m%.3f" % r for r in res))

print("gt spells:", [metrics(D[n]["y"], D[n]["y"])["n_gt_spells"] for n in D])
print("logit scale sensitivity")
for s in [0.3, 0.5, 0.7, 1.0, 1.5, 2.5]:
    show("scale %.1f" % s, evalset(lambda lg: lg * s))
print("D offset sensitivity")
for b in [-2, -1, -0.5, 0, 0.5, 1]:
    show("D %+.1f" % b, evalset(lambda lg: lg + np.array([0, 0, b])))
print("raw argmax (no decoder)")
for n in D:
    d = D[n]; print(n, metrics(d["y"], d["logits"].argmax(1)))
```