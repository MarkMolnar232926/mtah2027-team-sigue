"""Sequence model + decoder.

PossessionNet: per-frame features -> per-frame log-probabilities over (H, A, D), bidirectional,
so every frame sees the whole surrounding context (a possession is an interval, not a frame).

viterbi(): turns those probabilities into a clean label sequence. Each change of state costs a
penalty, which is what stops the output from flickering into hundreds of fake possessions -- the
single thing the ranking metric punishes hardest. Penalties are tuned on held-out DFL matches only.
"""
import numpy as np
import torch
import torch.nn as nn


class PossessionNet(nn.Module):
    def __init__(self, n_in, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(hidden, hidden, 9, padding=4)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden + hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, 3))

    def forward(self, x):                       # x (B, T, F)
        h = self.inp(x)
        h = h + torch.relu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        g, _ = self.gru(h)
        return self.head(torch.cat([g, h], -1))  # logits (B, T, 3)


@torch.no_grad()
def predict(model, X, device, chunk=6000, overlap=1000):
    """Log-probs for one long sequence, with overlapping windows stitched at their centres."""
    model.eval()
    T = len(X)
    out = np.zeros((T, 3), np.float32)
    step = chunk - 2 * overlap
    start = 0
    while start < T:
        a = max(0, start - overlap)
        b = min(T, start + step + overlap)
        x = torch.from_numpy(np.asarray(X[a:b], np.float32))[None].to(device)
        lp = torch.log_softmax(model(x)[0], -1).cpu().numpy()
        out[start:min(T, start + step)] = lp[start - a:start - a + min(step, T - start)]
        start += step
    return out


def viterbi(logp, switch_ha, switch_alive):
    """logp (T,3) for H, A, D. Cost switch_ha for H<->A, switch_alive for alive<->dead."""
    T = len(logp)
    trans = np.array([[0, -switch_ha, -switch_alive],
                      [-switch_ha, 0, -switch_alive],
                      [-switch_alive, -switch_alive, 0]], np.float64)
    score = logp[0].astype(np.float64).copy()
    back = np.zeros((T, 3), np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans          # from i to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.empty(T, np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
