"""Turn DFL's per-touch BallPossession into the challenge's *controlled* possession.

CHALLENGE.md: a change of possession needs controlled possession; a touch without control, a
deflection or an aerial duel without control does not count, and while the ball is loose,
possession stays with the last team that had it. A restart always has a clear possessor.

Rule implemented (a definition, not fitted to any ground truth): inside an alive stretch, a team
spell shorter than `min_control` seconds that does not start the stretch is not a change of
possession -- it is relabelled to the team that had the ball before it.
Codes: 0 = H, 1 = A, 2 = D.
"""
import numpy as np

FPS = 25


def raw_codes(status, poss):
    return np.where(status == 0, 2, np.where(poss == 1, 0, 1)).astype(np.int8)


def controlled(codes, min_control=2.0):
    out = codes.copy()
    n = len(out)
    minf = int(round(min_control * FPS))
    i = 0
    while i < n:
        if out[i] == 2:
            i += 1
            continue
        j = i
        while j < n and out[j] != 2:
            j += 1
        seg = out[i:j]          # one alive stretch
        cur = seg[0]            # the restart's possessor is always valid
        k = 0
        while k < len(seg):
            m = k
            while m < len(seg) and seg[m] == seg[k]:
                m += 1
            if seg[k] != cur and (m - k) < minf:
                seg[k:m] = cur
            else:
                cur = seg[k]
            k = m
        i = j
    return out
