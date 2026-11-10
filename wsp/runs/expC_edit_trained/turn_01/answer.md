Baseline first: one `score()` to see per-source spell counts, confusions and the argmax-vs-decoded gap, which tells me where the decoder/logit scale interacts with the network.

```python
base_score = score()
for src, d in base_score.items():
    print(src)
    for k, v in d.items():
        print("   ", k, v if not isinstance(v, np.ndarray) else v.tolist())
print(passes_left())
```