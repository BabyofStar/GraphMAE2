"""Real PyTorch/DGL tests, to run in the verified graphmae2 server environment."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import dgl

from models.adaptive_masking import AdaptiveInputMask
from models.edcoder import PreModel
from models.input_masking import StructureInputMask
from models.masking_math import proportional_quotas


def devices():
    return [torch.device("cpu")] + ([torch.device("cuda:0")]
                                    if torch.cuda.is_available() else [])


def rng():
    return [torch.get_rng_state().clone()] + [s.clone() for s in torch.cuda.get_rng_state_all()]


def reset_rng(states):
    torch.set_rng_state(states[0])
    if len(states) > 1:
        torch.cuda.set_rng_state_all(states[1:])


def graph(device, n=12):
    src = list(range(n)) + [0, 0, 2, 4]
    dst = [(i + 1) % n for i in range(n)] + [3, 6, 5, 8]
    g = dgl.add_self_loop(dgl.to_bidirected(dgl.graph((src, dst), num_nodes=n))).to(device)
    g.ndata["feat"] = (torch.arange(n * 8, dtype=torch.float32,
                                    device=device).reshape(n, 8) % 11) / 11
    g.ndata["label"] = torch.arange(n, device=device) % 2
    return g


def kwargs():
    return dict(in_dim=8, num_hidden=16, num_layers=2, num_dec_layers=1,
                num_remasking=3, nhead=2, nhead_out=1, activation="prelu",
                feat_drop=0.2, attn_drop=0.1, negative_slope=0.2, residual=False,
                norm=None, mask_rate=0.5, remask_rate=0.5, remask_method="fixed",
                encoder_type="gat", decoder_type="gat", loss_fn="sce", alpha_l=2,
                lam=0.15, momentum=0.995, zero_init=True)


class AdaptiveSamplingTests(unittest.TestCase):
    def test_preparation_has_no_parameters_checkpoint_keys_or_rng_changes(self):
        for device in devices():
            for mode in ("random", "structure_curriculum") + AdaptiveInputMask.NEW_MODES:
                with self.subTest(device=str(device), mode=mode):
                    g = graph(device)
                    before = rng()
                    sampler = AdaptiveInputMask(mode, diagnostics=True).to(device)
                    sampler.prepare(g, device, g.ndata["feat"])
                    for a, b in zip(before, rng()):
                        self.assertTrue(torch.equal(a, b))
                    self.assertEqual(list(sampler.parameters()), [])
                    self.assertEqual(dict(sampler.state_dict()), {})

    def test_labels_and_split_masks_do_not_affect_density(self):
        g = graph("cpu")
        first = AdaptiveInputMask("density_stratified")
        first.prepare(g, "cpu", g.ndata["feat"])
        g.ndata["label"].fill_(999)
        for name in ("train_mask", "val_mask", "test_mask"):
            g.ndata[name] = torch.zeros(g.num_nodes(), dtype=torch.bool)
        second = AdaptiveInputMask("density_stratified")
        second.prepare(g, "cpu", g.ndata["feat"])
        np.testing.assert_array_equal(first._density, second._density)
        np.testing.assert_array_equal(first._membership, second._membership)

    def test_legacy_sampling_and_rng_match_v1_for_all_modes(self):
        for device in devices():
            for mode in StructureInputMask.MODES:
                for diagnostics in (False, True):
                    with self.subTest(device=str(device), mode=mode, diagnostics=diagnostics):
                        g = graph(device)
                        old = StructureInputMask(mode).to(device)
                        new = AdaptiveInputMask(mode, diagnostics=diagnostics).to(device)
                        old.prepare(g, device)
                        new.prepare(g, device, g.ndata["feat"])
                        for step in (0, 5, 10):
                            old.set_progress(step, 11)
                            new.set_progress(step, 11)
                            state = rng()
                            expected = old.permutation(g, device, 0.5)
                            after = rng()
                            reset_rng(state)
                            actual = new.permutation(g, device, 0.5)
                            self.assertTrue(torch.equal(expected, actual))
                            for a, b in zip(after, rng()):
                                self.assertTrue(torch.equal(a, b))

    def test_new_modes_have_exact_count_unique_nodes_and_density_quotas(self):
        for device in devices():
            g = graph(device, 11)
            for mode in AdaptiveInputMask.NEW_MODES:
                sampler = AdaptiveInputMask(mode, log_interval=0).to(device)
                sampler.prepare(g, device, g.ndata["feat"])
                for step in (0, 5, 10):
                    sampler.set_progress(step, 11)
                    for rate in (0.0, 0.5, 0.9, 1.0):
                        with self.subTest(device=str(device), mode=mode, step=step, rate=rate):
                            perm = sampler.permutation(g, device, rate)
                            self.assertEqual(perm.unique().numel(), 11)
                            self.assertEqual(perm.numel(), 11)
                            k = int(rate * 11)
                            if mode in sampler.STRATIFIED and k:
                                q = proportional_quotas([len(p) for p in sampler._partitions], k)
                                ids = perm[:k].cpu().numpy()
                                self.assertEqual(np.bincount(sampler._membership[ids],
                                                            minlength=len(q)).tolist(), q)

    def test_difficulty_zero_alpha_and_warmup_use_original_randperm(self):
        for alpha, step in ((0.5, 0), (0.5, 2), (0.0, 10)):
            sampler = AdaptiveInputMask("reconstruction_curriculum", alpha_max=alpha)
            g = graph("cpu")
            sampler.prepare(g, "cpu")
            sampler.set_progress(step, 11)
            state = rng()
            expected = torch.randperm(12)
            after = rng()
            reset_rng(state)
            self.assertTrue(torch.equal(sampler.permutation(g, "cpu", 0.5), expected))
            self.assertTrue(torch.equal(after[0], rng()[0]))

    def test_ema_updates_only_observed_nodes_and_has_no_autograd_graph(self):
        sampler = AdaptiveInputMask("reconstruction_curriculum", ema_beta=0.5, log_interval=0)
        sampler.prepare(graph("cpu"), "cpu")
        ids = torch.tensor([0, 2])
        target = torch.tensor([[1., 0.], [1., 0.]])
        first = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        sampler.set_progress(0, 3)
        before = rng()
        sampler.observe_reconstruction(ids, target, first, "sce", 1)
        torch.testing.assert_close(sampler.error_ema[ids], torch.tensor([0., 1.]))
        sampler.set_progress(1, 3)
        sampler.observe_reconstruction(ids, target, first.flip(0), "sce", 1)
        torch.testing.assert_close(sampler.error_ema[ids], torch.tensor([0.5, 0.5]))
        self.assertEqual(sampler.observation_count[ids].tolist(), [2, 2])
        self.assertEqual(int(sampler.observation_count.sum()), 4)
        self.assertIsNone(first.grad)
        self.assertFalse(sampler.error_ema.requires_grad)
        self.assertIsNone(sampler.error_ema.grad_fn)
        for a, b in zip(before, rng()):
            self.assertTrue(torch.equal(a, b))

    def test_scores_clip_outlier_and_give_unseen_nodes_positive_neutral_weight(self):
        sampler = AdaptiveInputMask("reconstruction_curriculum", clip_quantile=0.5)
        sampler.prepare(graph("cpu"), "cpu")
        sampler.error_ema[:3] = torch.tensor([1., 2., 1000.])
        sampler.observation_count[:3] = 1
        scores = sampler.difficulty_scores()
        self.assertEqual(float(scores[2]), 2.)
        self.assertEqual(float(scores[3]), 2.)
        self.assertTrue((scores > 0).all().item())

    def test_large_errors_actually_bias_later_sampling(self):
        sampler = AdaptiveInputMask("reconstruction_curriculum", alpha_max=1., log_interval=0)
        g = graph("cpu")
        sampler.prepare(g, "cpu")
        sampler.set_progress(0, 11)
        ids = torch.arange(12)
        target = torch.tensor([[1., 0.]]).repeat(12, 1)
        recon = torch.tensor([[1., 0.]]).repeat(12, 1)
        recon[6:] = torch.tensor([0., 1.])
        sampler.observe_reconstruction(ids, target, recon, "sce", 1)
        sampler.set_progress(10, 11)
        torch.manual_seed(812)
        hard_counts = [int((sampler.permutation(g, "cpu", 1 / 3)[:4] >= 6).sum())
                       for _ in range(100)]
        self.assertGreater(np.mean(hard_counts), 3.5)

    def test_duplicate_feedback_and_nonfinite_errors_fail(self):
        sampler = AdaptiveInputMask("reconstruction_curriculum", log_interval=0)
        sampler.prepare(graph("cpu"), "cpu")
        sampler.set_progress(0, 3)
        ids, value = torch.tensor([0]), torch.ones(1, 2)
        sampler.observe_reconstruction(ids, value, value)
        with self.assertRaisesRegex(RuntimeError, "one feedback"):
            sampler.observe_reconstruction(ids, value, value)
        sampler.set_progress(1, 3)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            sampler.observe_reconstruction(ids, value, value * float("nan"))

    def test_statistics_export_contains_arrays_not_pickle_and_never_overwrites(self):
        sampler = AdaptiveInputMask("density_stratified", diagnostics=True)
        g = graph("cpu")
        sampler.prepare(g, "cpu", g.ndata["feat"])
        sampler.set_progress(0, 1)
        ids = sampler.permutation(g, "cpu", 0.5)[:6]
        value = g.ndata["feat"][ids]
        sampler.observe_reconstruction(ids, value, value)
        self.assertEqual(sum(x["masked"] for x in sampler.last_feedback["density_layers"]), 6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nodes.npz"
            before = rng()
            sampler.export_node_statistics(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["observation_count"].sum(), 6)
                self.assertEqual(data["density_layer"].shape, (12,))
            for a, b in zip(before, rng()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaises(FileExistsError):
                sampler.export_node_statistics(path)

    def test_invalid_hyperparameters_and_missing_features_fail_early(self):
        for options in ({"ema_beta": 1.}, {"ema_beta": float("nan")},
                        {"clip_quantile": 0.}, {"density_bins": True},
                        {"density_chunk_size": 0}):
            with self.assertRaises(ValueError):
                AdaptiveInputMask(**options)
        with self.assertRaisesRegex(ValueError, "original node features"):
            AdaptiveInputMask("density_stratified").prepare(graph("cpu"), "cpu")


class AdaptiveModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        path = Path(__file__).parent / "fixtures" / "edcoder_v1_2_reference.py"
        spec = importlib.util.spec_from_file_location("models._v12_reference", path)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def train_tiny(self, factory, g, mode, diagnostics=False, alpha=0.5):
        options = dict(kwargs(), input_mask_mode=mode, mask_log_interval=0,
                       structure_alpha_max=alpha)
        if factory is PreModel:
            options["input_mask_diagnostics"] = diagnostics
        torch.manual_seed(125)
        with contextlib.redirect_stdout(io.StringIO()):
            model = factory(**options).to(g.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.prepare_input_mask(g, g.device)
        losses = []
        for step in range(5):
            model.set_input_mask_progress(step, 5)
            loss = model(g, g.ndata["feat"], targets=torch.arange(g.num_nodes(), device=g.device))
            losses.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model, torch.stack(losses), rng()

    def test_legacy_full_model_losses_weights_and_rng_match_pinned_v12(self):
        for device in devices():
            for mode in StructureInputMask.MODES:
                with self.subTest(device=str(device), mode=mode):
                    g = graph(device)
                    old, old_losses, old_rng = self.train_tiny(self.reference.PreModel, g, mode)
                    new, new_losses, new_rng = self.train_tiny(PreModel, g, mode, True)
                    torch.testing.assert_close(old_losses, new_losses, rtol=1e-4, atol=1e-6)
                    self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                    for key, value in old.state_dict().items():
                        torch.testing.assert_close(value, new.state_dict()[key], rtol=1e-4, atol=1e-6)
                    for a, b in zip(old_rng, new_rng):
                        self.assertTrue(torch.equal(a, b))

    def test_new_modes_real_forward_backward_feedback_and_cpu_reload(self):
        for device in devices():
            for mode in AdaptiveInputMask.NEW_MODES:
                with self.subTest(device=str(device), mode=mode):
                    trained, losses, _ = self.train_tiny(PreModel, graph(device), mode, True)
                    self.assertTrue(torch.isfinite(losses).all().item())
                    self.assertEqual(int(trained.input_masker.observation_count.sum()), 5 * 6)
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                                        for p in trained.encoder.parameters()))
                    for parameter in trained.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    with contextlib.redirect_stdout(io.StringIO()):
                        fresh = PreModel(**dict(kwargs(), input_mask_mode=mode))
                    fresh.load_state_dict({k: v.cpu() for k, v in trained.state_dict().items()}, strict=True)
                    self.assertEqual(fresh.input_masker.error_ema.numel(), 0)

    def test_zero_strength_difficulty_matches_original_random_model(self):
        for device in devices():
            g = graph(device)
            old, old_losses, old_rng = self.train_tiny(self.reference.PreModel, g, "random")
            new, new_losses, new_rng = self.train_tiny(
                PreModel, g, "reconstruction_curriculum", True, alpha=0.0)
            torch.testing.assert_close(old_losses, new_losses, rtol=1e-4, atol=1e-6)
            for name, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[name], rtol=1e-4, atol=1e-6)
            for a, b in zip(old_rng, new_rng):
                self.assertTrue(torch.equal(a, b))

    def test_random_decoder_remasking_is_rejected_for_new_feedback(self):
        with self.assertRaisesRegex(ValueError, "fixed decoder"):
            PreModel(**dict(kwargs(), remask_method="random",
                            input_mask_mode="reconstruction_curriculum"))

    def test_cli_factory_preserves_new_nondefault_hyperparameters_after_yaml(self):
        import sys
        from unittest import mock
        from models import build_model
        from utils import build_args, load_best_configs
        argv = ["test", "--dataset", "cora", "--use_cfg", "--input_mask_mode",
                "reconstruction_density_curriculum", "--input_mask_diagnostics",
                "--difficulty_ema_beta", "0.8", "--difficulty_clip_quantile", "0.7",
                "--density_bins", "3", "--density_chunk_size", "16"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            args = load_best_configs(build_args())
            args.num_features, args.num_hidden, args.num_heads = 8, 16, 2
            sampler = build_model(args).input_masker
        self.assertEqual(args.max_epoch, 2000)
        self.assertEqual(args.remask_method, "fixed")
        self.assertEqual(sampler.mode, "reconstruction_density_curriculum")
        self.assertTrue(sampler.diagnostics)
        self.assertEqual((sampler.ema_beta, sampler.clip_quantile), (0.8, 0.7))
        self.assertEqual((sampler.density_bins, sampler.density_chunk_size), (3, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
