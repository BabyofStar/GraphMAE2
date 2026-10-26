"""Run from the repository root: python -m unittest discover -s tests -p test_input_masking.py -v"""

import subprocess
import types
import unittest
from unittest import mock
import sys

import dgl
import torch

from models.edcoder import PreModel
from models.input_masking import StructureInputMask
from models import build_model
from utils import build_args, load_best_configs

BASE_COMMIT = "bc473e8b750d6e90b7cba1c6bb20d858f617d90c"


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available():
        result.append(torch.device("cuda", 0))
    return result


def rng_state():
    return (torch.get_rng_state().clone(),
            [s.clone() for s in torch.cuda.get_rng_state_all()])


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def assert_rng_equal(test, first, second):
    test.assertTrue(torch.equal(first[0], second[0]), "CPU RNG changed")
    test.assertEqual(len(first[1]), len(second[1]))
    for a, b in zip(first[1], second[1]):
        test.assertTrue(torch.equal(a, b), "CUDA RNG changed")


def graph_on(device):
    g = dgl.graph(([0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]), num_nodes=4)
    return dgl.add_self_loop(g).to(device)


def model_kwargs():
    return dict(
        in_dim=8, num_hidden=16, num_layers=2, num_dec_layers=1,
        num_remasking=3, nhead=2, nhead_out=1, activation="prelu",
        feat_drop=0.2, attn_drop=0.1, negative_slope=0.2, residual=False,
        norm=None, mask_rate=0.5, remask_rate=0.5, remask_method="fixed",
        encoder_type="gat", decoder_type="gat", loss_fn="sce", alpha_l=2,
        lam=0.15, momentum=0.995, zero_init=True,
    )


class SamplingTests(unittest.TestCase):
    def test_default_is_disabled_and_has_no_checkpoint_keys(self):
        state = rng_state()
        masker = StructureInputMask()
        self.assertFalse(masker.enabled)
        self.assertEqual(len(masker.state_dict()), 0)
        self.assertEqual(len(list(masker.parameters())), 0)
        assert_rng_equal(self, state, rng_state())

    def test_score_excludes_self_loops_and_does_not_use_labels(self):
        for device in devices():
            with self.subTest(device=str(device)):
                g = dgl.graph(([0, 1, 1, 2, 2], [1, 0, 1, 2, 2]), num_nodes=4).to(device)
                g.ndata["label"] = torch.full((4,), 123, device=device)
                masker = StructureInputMask("structure_static").to(device)
                before = rng_state()
                masker.prepare(g, device)
                assert_rng_equal(self, before, rng_state())
                expected = torch.tensor([1., 1., 0., 0.], device=device)
                torch.testing.assert_close(masker.degree, expected)
                raw = torch.log1p(expected) + 1e-8
                torch.testing.assert_close(masker.score_weights, raw / raw.sum())
                self.assertTrue(torch.all(masker.score_weights > 0).item())
                before_score = masker.score_weights.clone()
                g.ndata["label"].fill_(-999)
                masker.prepare(g, device)
                torch.testing.assert_close(before_score, masker.score_weights)

    def test_partition_count_reproducibility_and_extreme_rates(self):
        for device in devices():
            for mode in ("structure_static", "structure_curriculum"):
                with self.subTest(device=str(device), mode=mode):
                    g = graph_on(device)
                    masker = StructureInputMask(mode, log_interval=0).to(device)
                    masker.prepare(g, device)
                    masker.set_progress(8, 11)
                    for rate in (0.0, 0.25, 0.5, 0.9, 1.0):
                        state = rng_state()
                        perm = masker.permutation(g, device, rate)
                        after = rng_state()
                        restore_rng(state)
                        repeated = masker.permutation(g, device, rate)
                        self.assertTrue(torch.equal(perm, repeated))
                        assert_rng_equal(self, after, rng_state())
                        self.assertEqual(perm.device, device)
                        self.assertEqual(perm.dtype, torch.int64)
                        self.assertEqual(perm[:int(rate * 4)].numel(), int(rate * 4))
                        self.assertTrue(torch.equal(
                            perm.sort().values, torch.arange(4, device=device)))

    def test_zero_alpha_and_curriculum_warmup_keep_randperm_rng(self):
        for device in devices():
            for mode, alpha, step in (
                ("random", 0.5, 0),
                ("structure_static", 0.0, 7),
                ("structure_curriculum", 0.0, 7),
                ("structure_curriculum", 0.5, 0),
            ):
                with self.subTest(device=str(device), mode=mode, alpha=alpha, step=step):
                    g = graph_on(device)
                    masker = StructureInputMask(mode, alpha_max=alpha).to(device)
                    masker.prepare(g, device)
                    masker.set_progress(step, 11)
                    state = rng_state()
                    expected = torch.randperm(4, device=device)
                    after = rng_state()
                    restore_rng(state)
                    actual = masker.permutation(g, device, .5)
                    self.assertTrue(torch.equal(actual, expected))
                    assert_rng_equal(self, after, rng_state())

    def test_schedule_boundaries_monotonicity_and_missing_progress(self):
        masker = StructureInputMask("structure_curriculum", alpha_max=.5)
        with self.assertRaises(RuntimeError):
            masker.effective_alpha()
        values = []
        for step in range(11):
            masker.set_progress(step, 11)
            values.append(masker.effective_alpha())
        self.assertAlmostEqual(values[0], 0.)
        self.assertAlmostEqual(values[2], 0.)
        self.assertAlmostEqual(values[5], .25)
        self.assertAlmostEqual(values[8], .5)
        self.assertAlmostEqual(values[10], .5)
        self.assertEqual(values, sorted(values))
        masker.set_progress(0, 1)
        self.assertEqual(masker.effective_alpha(), 0.)

    def test_empty_edge_graph_and_statistics(self):
        for device in devices():
            with self.subTest(device=str(device)):
                g = dgl.graph(([], []), num_nodes=5).to(device)
                masker = StructureInputMask("structure_static", log_interval=1).to(device)
                masker.prepare(g, device)
                torch.testing.assert_close(
                    masker.score_weights, torch.full((5,), .2, device=device))
                masker.set_progress(0, 1)
                perm = masker.permutation(g, device, .5)
                self.assertEqual(perm.unique().numel(), 5)
                self.assertEqual(masker.last_stats["masked"], 2)
                self.assertAlmostEqual(masker.last_stats["actual_mask_rate"], .4)

    def test_weighted_sampling_actually_favors_high_degree(self):
        g = dgl.graph(
            ([0] * 8 + list(range(1, 9)), list(range(1, 9)) + [0] * 8),
            num_nodes=9,
        )
        masker = StructureInputMask("structure_static", alpha_max=1., log_interval=0)
        masker.prepare(g, torch.device("cpu"))
        state = rng_state()
        torch.manual_seed(1234)
        counts = torch.zeros(9)
        for _ in range(1000):
            counts[masker.permutation(g, "cpu", 1 / 9)[0]] += 1
        restore_rng(state)
        self.assertGreater(counts[0].item(), 2 * counts[1:].mean().item())

    def test_invalid_parameters_and_unprepared_graph_fail(self):
        for kwargs in (
            dict(mode="not-a-mode"), dict(alpha_max=-.1), dict(alpha_max=1.1),
            dict(alpha_max=float("nan")), dict(warmup_ratio=.8, ramp_end_ratio=.2),
            dict(log_interval=-1),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StructureInputMask(**kwargs)
        masker = StructureInputMask("structure_static")
        with self.assertRaises(RuntimeError):
            masker.permutation(graph_on("cpu"), "cpu", .5)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = subprocess.run(
            ["git", "show", BASE_COMMIT + ":models/edcoder.py"],
            check=True, capture_output=True, encoding="utf-8",
        ).stdout
        module = types.ModuleType("models._original_edcoder_for_test")
        module.__package__ = "models"
        exec(compile(text, "<original-edcoder>", "exec"), module.__dict__)
        cls.OriginalPreModel = module.PreModel

    def test_cli_factory_and_official_yaml_keep_new_flags(self):
        argv = ["test", "--dataset", "cora", "--use_cfg",
                "--input_mask_mode", "structure_static",
                "--structure_alpha_max", "0.3", "--mask_log_interval", "0"]
        with mock.patch.object(sys, "argv", argv):
            args = load_best_configs(build_args())
        self.assertEqual(args.input_mask_mode, "structure_static")
        self.assertAlmostEqual(args.structure_alpha_max, .3)
        self.assertEqual(args.num_hidden, 1024)
        self.assertEqual(args.max_epoch, 2000)
        self.assertEqual(args.remask_method, "fixed")
        self.assertEqual(args.mask_rate, .5)
        # Use a small graph-model only for the factory wiring test.
        args.num_features, args.num_hidden, args.num_heads = 8, 16, 2
        model = build_model(args)
        self.assertEqual(model.input_masker.mode, "structure_static")
        self.assertAlmostEqual(model.input_masker.alpha_max, .3)

    def test_disabled_and_zero_strength_match_original_training_and_rng(self):
        for device in devices():
            for mode, strength in (("random", .5), ("random", 0.),
                                   ("structure_static", 0.), ("structure_curriculum", 0.)):
                with self.subTest(device=str(device), mode=mode, strength=strength):
                    torch.manual_seed(42)
                    original = self.OriginalPreModel(**model_kwargs()).to(device)
                    init_rng = rng_state()
                    torch.manual_seed(42)
                    changed = PreModel(
                        **model_kwargs(), input_mask_mode=mode,
                        structure_alpha_max=strength, mask_log_interval=0,
                    ).to(device)
                    assert_rng_equal(self, init_rng, rng_state())
                    original_keys = set(original.state_dict())
                    self.assertEqual(original_keys, set(changed.state_dict()))
                    for key, value in original.state_dict().items():
                        torch.testing.assert_close(value, changed.state_dict()[key])

                    g = graph_on(device)
                    x = torch.randn(4, 8, device=device)
                    targets = torch.arange(4, device=device)
                    changed.prepare_input_mask(g, device)
                    opt_old = torch.optim.Adam(original.parameters(), lr=.001)
                    opt_new = torch.optim.Adam(changed.parameters(), lr=.001)
                    for step in range(3):
                        state = rng_state()
                        original.train()
                        opt_old.zero_grad()
                        loss_old = original(g, x, targets=targets)
                        loss_old.backward()
                        opt_old.step()
                        after = rng_state()

                        restore_rng(state)
                        changed.train()
                        changed.set_input_mask_progress(step, 3)
                        opt_new.zero_grad()
                        loss_new = changed(g, x, targets=targets)
                        loss_new.backward()
                        opt_new.step()

                        assert_rng_equal(self, after, rng_state())
                        torch.testing.assert_close(loss_old, loss_new, rtol=1e-4, atol=1e-6)
                        for key, value in original.state_dict().items():
                            torch.testing.assert_close(
                                value, changed.state_dict()[key], rtol=1e-4, atol=1e-6)

    def test_positive_strength_backward_is_finite_and_progress_does_not_change_ema_epoch(self):
        for device in devices():
            for mode in ("structure_static", "structure_curriculum"):
                with self.subTest(device=str(device), mode=mode):
                    model = PreModel(
                        **model_kwargs(), delayed_ema_epoch=1,
                        input_mask_mode=mode, mask_log_interval=0,
                    ).to(device)
                    g = graph_on(device)
                    x = torch.randn(4, 8, device=device)
                    model.prepare_input_mask(g, device)
                    model.set_input_mask_progress(8, 11)
                    teacher_before = {
                        k: v.clone() for k, v in model.encoder_ema.state_dict().items()
                    }
                    with torch.no_grad():
                        next(model.encoder.parameters()).add_(.1)
                    loss = model(g, x, targets=torch.arange(4, device=device))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss).item())
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
                    for key, value in teacher_before.items():
                        torch.testing.assert_close(value, model.encoder_ema.state_dict()[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
