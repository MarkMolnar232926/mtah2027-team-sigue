The simulator reproduces the network (h exact; logits differ only through window-edge memory), and the four-view emulation works. Notable: on SC the role-swapped views are clearly less accurate (0.943 vs 0.975 team accuracy) — memorisation of the practice matches doesn't transfer to the swapped view, so swapped views are a rough proxy for held-out behaviour. Since I now only need `x` and `y` from a forward pass, I'll spend 9 passes to build a much larger offline dev set (spell statistics on ~45 possessions are too noisy), keeping 11 passes in reserve.

```python
for n in list(D):
    D[n] = {"x": D[n]["x"], "y": D[n]["y"]}
S = {}
def grab2(name, half, start, n=24000):
    o = forward(half, start=start, n=n)
    D[name] = {"x": o["x"].copy(), "y": o["y"].copy()}

grab2("dfl1a", 1, 0)
grab2("dfl0b", 0, 44000)
grab2("dfl3a", 3, 6000)
grab2("dfl3b", 3, 51259)
grab2("sc4", 4, 20000)
grab2("sc5", 5, 30000)
grab2("sc6", 6, 10000)
grab2("sc9", 9, 40000)
grab2("met11", 11, 20000)
print(passes_left(), {k: (v["x"].shape, int((v["y"] < 2).sum())) for k, v in D.items()})
NAMES = ["dfl0", "dfl2", "dfl1a", "dfl0b", "dfl3a", "dfl3b", "sc8", "sc4", "sc5", "sc6", "sc9", "met10", "met11"]
GROUPS = [NAMES[0:3], NAMES[3:6], NAMES[6:9], NAMES[9:11], NAMES[11:13]]
```