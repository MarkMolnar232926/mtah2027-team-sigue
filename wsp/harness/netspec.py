"""The network a weight-space-programming run writes into: PossessionNet(48, hidden=128).

It is the architecture of the trained models in models/*.pt (solution/model.py), unchanged, so a
hand-written checkpoint loads through eval_external.load_models and runs in eval_fold.py and
predict.py like any trained one.

The state is kept as numpy arrays under the PyTorch state-dict keys. ALIASES gives every block a
short name and, for the GRU and the head, one name per gate / per input group, so that nobody has
to compute row offsets inside the stacked PyTorch matrices. Every alias is a numpy *view*: writing
through it writes the state.

Orientation is PyTorch's everywhere: T[unit, input].
"""
import os

import numpy as np

N_FEAT = 48
HIDDEN = 128
KERNEL = 9
H = HIDDEN

# state-dict key -> shape
SHAPES = {
    "inp.0.weight": (H, N_FEAT), "inp.0.bias": (H,),
    "conv.weight": (H, H, KERNEL), "conv.bias": (H,),
    "head.0.weight": (H, 3 * H), "head.0.bias": (H,),
    "head.2.weight": (3, H), "head.2.bias": (3,),
}
for _layer, _n_in in ((0, H), (1, 2 * H)):
    for _suffix in ("", "_reverse"):
        SHAPES["gru.weight_ih_l%d%s" % (_layer, _suffix)] = (3 * H, _n_in)
        SHAPES["gru.weight_hh_l%d%s" % (_layer, _suffix)] = (3 * H, H)
        SHAPES["gru.bias_ih_l%d%s" % (_layer, _suffix)] = (3 * H,)
        SHAPES["gru.bias_hh_l%d%s" % (_layer, _suffix)] = (3 * H,)

N_PARAMS = int(sum(np.prod(s) for s in SHAPES.values()))

# alias -> (state-dict key, slice of the leading axis or None, slice of the second axis or None)
ALIASES = {
    "inp.W": ("inp.0.weight", None, None), "inp.b": ("inp.0.bias", None, None),
    "conv.W": ("conv.weight", None, None), "conv.b": ("conv.bias", None, None),
    "head1.Wf": ("head.0.weight", None, slice(0, H)),
    "head1.Wb": ("head.0.weight", None, slice(H, 2 * H)),
    "head1.Wh": ("head.0.weight", None, slice(2 * H, 3 * H)),
    "head1.b": ("head.0.bias", None, None),
    "head2.W": ("head.2.weight", None, None), "head2.b": ("head.2.bias", None, None),
}
# PyTorch stacks the GRU gates along the first axis in the order reset, update, new.
for _layer in (0, 1):
    for _suffix, _d in (("", "f"), ("_reverse", "b")):
        _name = "gru%d%s" % (_layer + 1, _d)
        for _g, _gate in enumerate("rzn"):
            _rows = slice(_g * H, (_g + 1) * H)
            ALIASES["%s.Wi%s" % (_name, _gate)] = ("gru.weight_ih_l%d%s" % (_layer, _suffix), _rows, None)
            ALIASES["%s.Wh%s" % (_name, _gate)] = ("gru.weight_hh_l%d%s" % (_layer, _suffix), _rows, None)
            ALIASES["%s.bi%s" % (_name, _gate)] = ("gru.bias_ih_l%d%s" % (_layer, _suffix), _rows, None)
            ALIASES["%s.bh%s" % (_name, _gate)] = ("gru.bias_hh_l%d%s" % (_layer, _suffix), _rows, None)


def zero_state():
    """Every weight 0 -- the starting point of a from-scratch run."""
    return {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}


def copy_state(state):
    return {k: v.copy() for k, v in state.items()}


def view(state, alias):
    """The numpy view an alias names. Writing into it writes the state."""
    key, rows, cols = ALIASES[alias]
    arr = state[key]
    if rows is not None:
        arr = arr[rows]
    if cols is not None:
        arr = arr[:, cols]
    return arr


def alias_shapes():
    z = zero_state()
    return {a: view(z, a).shape for a in ALIASES}


def summary(state):
    """Per state-dict tensor: how many weights are nonzero and the largest magnitude."""
    out = {}
    for k, v in state.items():
        out[k] = {"size": int(v.size), "nonzero": int(np.count_nonzero(v)),
                  "max_abs": float(np.abs(v).max()) if v.size else 0.0}
    return out


def n_nonzero(state):
    return int(sum(np.count_nonzero(v) for v in state.values()))


def from_checkpoint(path):
    """A models/*.pt checkpoint as a numpy state. Only the 48-feature, 128-unit shape is handled."""
    import torch
    ck = torch.load(path, map_location="cpu")
    if ck["n_feat"] != N_FEAT or ck["hidden"] != HIDDEN:
        raise ValueError("%s is %d features x %d hidden, expected %d x %d"
                         % (path, ck["n_feat"], ck["hidden"], N_FEAT, HIDDEN))
    state = {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in ck["state"].items()}
    if {k: v.shape for k, v in state.items()} != SHAPES:
        raise ValueError("%s does not have the PossessionNet(48, 128) tensors" % path)
    return state


def to_torch_state(state):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v, np.float32).copy()) for k, v in state.items()}


def save_checkpoint(state, path):
    """Same format as train.py writes: {"state", "n_feat", "hidden"}."""
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state": to_torch_state(state), "n_feat": N_FEAT, "hidden": HIDDEN}, path)


def save_sparse(state, path):
    """Compressed npz: tiny for a mostly-zero network, so every pass of a run can be kept."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **state)


def load_sparse(path):
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in SHAPES}
