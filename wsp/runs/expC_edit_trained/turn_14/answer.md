Baseline offline: DFL 0.705 (236 predicted vs 273 true possessions), SC 0.925, MET 1.0; swapped-only views are a bit worse on SC as expected. Now the timing question on the bigger set: event-aligned log-prob margin profiles around true turnovers and around dead↔live boundaries, plus the offset of the nearest matching predicted transition.

```python
def events(y):
    ch = np.flatnonzero(np.diff(y)) + 1
    return [(int(t), int(y[t - 1]), int(y[t])) for t in ch]

def timing(names, vs=(0, 1, 2, 3), win=75):
    prof = collections.defaultdict(list); offs = collections.defaultdict(list); found = collections.Counter(); tot = collections.Counter()
    for n in names:
        y = D[n]["y"][SL]; lp = pipe_views(LG0[n][:, SL], vs); pr = PRED0[(n, vs)]
        pev = events(pr)
        for t, a, b in events(y):
            if t < win or t > len(y) - win: continue
            kind = "turnover" if (a < 2 and b < 2) else ("restart D->live" if a == 2 else "stop live->D")
            if kind == "turnover":
                m = lp[t - win:t + win, b] - lp[t - win:t + win, a]
            elif a == 2:
                m = np.logaddexp(lp[t - win:t + win, 0], lp[t - win:t + win, 1]) - lp[t - win:t + win, 2]
            else:
                m = lp[t - win:t + win, 2] - np.logaddexp(lp[t - win:t + win, 0], lp[t - win:t + win, 1])
            prof[kind].append(m)
            tot[kind] += 1
            # nearest predicted transition of same type (a->b) within 3 s
            c = [tp - t for tp, pa, pb in pev if pa == a and pb == b and abs(tp - t) <= 75]
            if c:
                found[kind] += 1
                offs[kind].append(min(c, key=abs) / 25)
    for kind in sorted(prof):
        P = np.array(prof[kind]); med = np.median(P, 0); mean = P.mean(0)
        o = np.array(offs[kind])
        idx = [win - 50, win - 25, win - 12, win - 6, win - 3, win, win + 3, win + 6, win + 12, win + 25, win + 50]
        print("  %-16s n=%3d found %3d | offset median %+.2fs mean %+.2fs IQR [%+.2f, %+.2f] | frac late>0.4s %.2f early<-0.4s %.2f" % (
            kind, tot[kind], found[kind], np.median(o), o.mean(), np.quantile(o, .25), np.quantile(o, .75), (o > 0.4).mean(), (o < -0.4).mean()))
        print("      median margin(new-old) at t=-2,-1,-.5,-.24,-.12,0,+.12,+.24,+.5,+1,+2 s:", np.round(med[idx], 1))
        zc = np.flatnonzero(med > 0)
        print("      first frame where median margin > 0: %+.2fs ; frac of events with margin>0 at t=0: %.2f, at +0.5s: %.2f, at -0.5s: %.2f" % (
            (zc[0] - win) / 25 if len(zc) else 9, (P[:, win] > 0).mean(), (P[:, win + 12] > 0).mean(), (P[:, win - 12] > 0).mean()))

DFLN = [n for n in NAMES if n.startswith("dfl")]; SCN = [n for n in NAMES if n.startswith("sc")]; METN = [n for n in NAMES if n.startswith("met")]
print("DFL 4 views"); timing(DFLN)
print("SC 4 views"); timing(SCN)
print("SC swapped views only"); timing(SCN, (2, 3))
```