"""Unit tests for the weight-space-programming harness. No data needed.

    python -m unittest discover wsp/tests -v
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "wsp", "harness"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import edits  # noqa: E402
import netspec  # noqa: E402

H = netspec.HIDDEN


def block(body):
    return "design text\n\n```edits\n%s\n```\n" % body


class NetSpec(unittest.TestCase):
    def test_parameter_count_matches_the_trained_architecture(self):
        import torch  # noqa: F401
        from model import PossessionNet
        net = PossessionNet(netspec.N_FEAT, hidden=H)
        self.assertEqual(sum(p.numel() for p in net.parameters()), netspec.N_PARAMS)
        self.assertEqual({k: tuple(v.shape) for k, v in net.state_dict().items()}, netspec.SHAPES)

    def test_gate_aliases_point_at_the_right_rows(self):
        s = netspec.zero_state()
        netspec.view(s, "gru1f.Wiz")[5, 7] = 1.0
        netspec.view(s, "gru2b.Whn")[2, 3] = 2.0
        netspec.view(s, "gru2f.bir")[4] = 3.0
        netspec.view(s, "head1.Wb")[1, 6] = 4.0
        self.assertEqual(s["gru.weight_ih_l0"][H + 5, 7], 1.0)
        self.assertEqual(s["gru.weight_hh_l1_reverse"][2 * H + 2, 3], 2.0)
        self.assertEqual(s["gru.bias_ih_l1"][4], 3.0)
        self.assertEqual(s["head.0.weight"][1, H + 6], 4.0)
        self.assertEqual(netspec.n_nonzero(s), 4)

    def test_alias_shapes(self):
        shapes = netspec.alias_shapes()
        self.assertEqual(shapes["inp.W"], (H, 48))
        self.assertEqual(shapes["conv.W"], (H, H, 9))
        self.assertEqual(shapes["gru1b.Win"], (H, H))
        self.assertEqual(shapes["gru2f.Wir"], (H, 2 * H))
        self.assertEqual(shapes["head1.Wh"], (H, H))
        self.assertEqual(shapes["head2.W"], (3, H))

    def test_zero_network_runs_and_outputs_zero(self):
        import torch
        from model import PossessionNet
        net = PossessionNet(netspec.N_FEAT, hidden=H)
        net.load_state_dict(netspec.to_torch_state(netspec.zero_state()), strict=True)
        out = net.eval()(torch.randn(1, 50, 48))
        self.assertEqual(tuple(out.shape), (1, 50, 3))
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_checkpoint_round_trip(self):
        import tempfile
        s = netspec.zero_state()
        netspec.view(s, "inp.W")[0, 47] = -2.5
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hand.pt")
            netspec.save_checkpoint(s, path)
            back = netspec.from_checkpoint(path)
            netspec.save_sparse(s, os.path.join(d, "p.npz"))
            sparse = netspec.load_sparse(os.path.join(d, "p.npz"))
        for k in s:
            np.testing.assert_array_equal(s[k], back[k])
            np.testing.assert_array_equal(s[k], sparse[k])


class Edits(unittest.TestCase):
    def run_block(self, body, **kw):
        s = netspec.zero_state()
        return s, edits.apply_text(s, block(body), **kw)

    def test_scalar_range_list_and_inplace(self):
        s, r = self.run_block(
            "inp.W[3, 47] = -2.5   # one weight\n"
            "conv.W[10, 3, :] = 0.5\n"
            "head2.W[0, 0:3] = [4, -4, 0.5]\n"
            "gru1f.bhz[0:8] += 2\n"
            "gru1f.bhz[0:8:2] *= 3\n"
            "inp.b[-1] -= 1\n")
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["applied"], 6)
        self.assertEqual(s["inp.0.weight"][3, 47], -2.5)
        np.testing.assert_array_equal(s["conv.weight"][10, 3], np.full(9, 0.5, np.float32))
        np.testing.assert_array_equal(s["head.2.weight"][0, :3], [4, -4, 0.5])
        np.testing.assert_array_equal(s["gru.bias_hh_l0"][H:H + 8], [6, 2, 6, 2, 6, 2, 6, 2])
        self.assertEqual(s["inp.0.bias"][-1], -1)

    def test_nested_list_over_several_lines(self):
        s, r = self.run_block("head2.W[0:2, 0:2] = [[1, 2],\n   [3, 4]]\ninp.b[0] = 1")
        self.assertEqual(r["errors"], [])
        np.testing.assert_array_equal(s["head.2.weight"][:2, :2], [[1, 2], [3, 4]])
        self.assertEqual(s["inp.0.bias"][0], 1)

    def test_bad_statements_are_skipped_and_reported(self):
        s, r = self.run_block(
            "nope.W[0] = 1\n"                    # unknown tensor
            "inp.W[200, 0] = 1\n"                # index out of range
            "inp.W[0, 0:99] = 1\n"               # range out of bounds
            "inp.W[0, 0:3] = [1, 2]\n"           # shape mismatch
            "inp.W[0, 0] = __import__('os')\n"   # not a number
            "inp.W[0, 0] = 1e9\n"                # too large
            "inp.W[1+1, 0] = 1\n"                # arithmetic in an index
            "inp.W[0, 1] = 7\n")
        self.assertEqual(r["statements"], 8)
        self.assertEqual(r["applied"], 1)
        self.assertEqual(len(r["errors"]), 7)
        self.assertEqual(netspec.n_nonzero(s), 1)
        self.assertEqual(s["inp.0.weight"][0, 1], 7)

    def test_only_the_last_block_is_read(self):
        s = netspec.zero_state()
        text = block("inp.b[0] = 1") + "second thoughts\n" + block("inp.b[1] = 2")
        r = edits.apply_text(s, text)
        self.assertEqual((s["inp.0.bias"][0], s["inp.0.bias"][1]), (0, 2))
        self.assertEqual(r["applied"], 1)

    def test_no_block(self):
        s = netspec.zero_state()
        r = edits.apply_text(s, "no weights here")
        self.assertFalse(r["found_block"])
        self.assertEqual(len(r["errors"]), 1)

    def test_reset_and_restore(self):
        saved = netspec.zero_state()
        saved["inp.0.bias"][5] = 9
        s = netspec.zero_state()
        s["inp.0.bias"][0] = 1
        r = edits.apply_text(s, block("restore 2\ninp.b[1] = 3"), reset=netspec.zero_state,
                             restore=lambda n: saved if n == 2 else None)
        self.assertEqual(r["errors"], [])
        self.assertEqual(list(s["inp.0.bias"][[0, 1, 5]]), [0, 3, 9])
        r = edits.apply_text(s, block("restore 7\nreset\ninp.b[2] = 4"), reset=netspec.zero_state,
                             restore=lambda n: None)
        self.assertEqual(len(r["errors"]), 1)
        self.assertEqual(netspec.n_nonzero(s), 1)
        self.assertEqual(s["inp.0.bias"][2], 4)

    def test_an_edited_network_loads_and_runs(self):
        import torch
        from model import PossessionNet
        s, r = self.run_block("inp.W[0, 47] = -3\ninp.b[0] = 0.5\nhead1.Wh[0, 0] = 2\n"
                              "head2.W[0, 0] = 5\nhead2.b[2] = -1")
        self.assertEqual(r["errors"], [])
        net = PossessionNet(netspec.N_FEAT, hidden=H)
        net.load_state_dict(netspec.to_torch_state(s), strict=True)
        x = torch.zeros(1, 4, 48)
        x[0, :, 47] = -1.0                      # home player much closer to the ball
        out = net.eval()(x)[0, 0]
        self.assertGreater(float(out[0]), float(out[1]))
        self.assertAlmostEqual(float(out[2]), -1.0, places=5)


class BatchedInference(unittest.TestCase):
    def test_batched_predict_matches_the_stock_predict(self):
        import torch
        import evaluate
        from model import PossessionNet, predict
        torch.manual_seed(0)
        net = PossessionNet(netspec.N_FEAT, hidden=H).eval()
        rng = np.random.default_rng(0)
        Xs = [rng.standard_normal((1530, 48)).astype(np.float32) for _ in range(2)]
        got = evaluate.predict_batched(net, Xs, chunk=600, overlap=100)
        for X, lp in zip(Xs, got):
            np.testing.assert_allclose(lp, predict(net, X, "cpu", chunk=600, overlap=100), atol=2e-4)


class Guard(unittest.TestCase):
    def test_ordinary_analysis_code_is_accepted(self):
        import guard
        code = ("import numpy as np\nfrom collections import Counter\n"
                "def f(v):\n    return float(np.abs(v).mean())\n"
                "u, s, vt = np.linalg.svd(W['head2.W'], full_matrices=False)\n"
                "W['head2.b'][2] += 0.1\nprint(f(s), Counter([1, 1, 2]))\n")
        ns = {"__builtins__": guard.safe_builtins(), "W": {"head2.W": np.eye(3), "head2.b": np.zeros(3)}}
        exec(guard.check(code), ns)
        self.assertAlmostEqual(ns["W"]["head2.b"][2], 0.1)

    def test_escapes_are_refused(self):
        import guard
        for code in (
            "import os",
            "import numpy.ctypeslib",
            "from os import system",
            "open('x', 'w')",
            "().__class__.__bases__",
            "np.load('x.npy')",
            "np.save('x.npy', W)",
            "W.tofile('x')",
            "eval('1+1')",
            "getattr(np, 'load')",
            "__import__('os')",
            "x = [c for c in ().__class__.__mro__]",
            "import subprocess as numpy",
            "def _f():\n    pass",
            "class A:\n    pass",
            "type(W)",
        ):
            with self.assertRaises(guard.Refused, msg=code):
                guard.check(code)

    def test_runtime_import_is_also_blocked(self):
        import guard
        with self.assertRaises(ImportError):
            guard.safe_builtins()["__import__"]("os")


class PromptBuild(unittest.TestCase):
    def test_prompts_build_and_name_every_tensor(self):
        import build_prompt
        import kernel
        for text in (build_prompt.build(0), build_prompt.build(10),
                     build_prompt.build_edit(25, 40, 20, kernel.OUTPUT_CAP, kernel.API_DOC)):
            self.assertNotIn("{", text.split("```python")[0])       # no unfilled placeholder
            for name in ("inp.W", "conv.W", "gru1f.Wiz", "gru2b.Whn", "head1.Wh", "head2.b"):
                self.assertIn(name, text)
        self.assertIn("restore 3", build_prompt.build(10))
        self.assertNotIn("restore 3", build_prompt.build(0))


if __name__ == "__main__":
    unittest.main()
