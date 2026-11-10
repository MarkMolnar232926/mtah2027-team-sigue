I'll start with free inspection: the practice halves, the pass budget, and some basic weight statistics before spending any passes.

```python
print(passes_left())
hs = halves()
print(hs)
for k in W:
    w = W[k]
    print(k, w.shape, float(np.abs(w).mean()), float(np.abs(w).max()))
print("head2.b", W["head2.b"])
# relative size of the three input paths into m
for k in ["head1.Wf", "head1.Wb", "head1.Wh"]:
    print(k, "fro", float(np.linalg.norm(W[k])))
```