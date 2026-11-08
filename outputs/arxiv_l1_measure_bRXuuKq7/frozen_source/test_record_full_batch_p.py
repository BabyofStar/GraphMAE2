"""v2 recorder contract tests plus one real-model recording regression."""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import types
import unittest

import record_full_batch_p as recorded


class V2RecorderTests(unittest.TestCase):
    def test_four_explicit_methods(self):
        self.assertEqual(recorded.METHODS, {"S":"density_stratified", "H":"rank_difficulty",
                                          "P":"rank_progress", "PS":"rank_progress_shuffled"})

    def test_formal_and_outside_development_seeds_are_rejected(self):
        for phase, seed in (("formal", 10), ("development", 3), ("smoke", 1), ("diagnostic", 1)):
            with self.assertRaises(RuntimeError):
                recorded.validate_development_cli(argparse.Namespace(
                    phase=phase, seed=seed, method="P"))
        recorded.validate_development_cli(argparse.Namespace(
            phase="development", seed=2, method="PS"))

    def test_official_configuration_is_loaded_before_smoke_epoch_override(self):
        import sys
        from unittest import mock
        def build():
            self.assertIn("--input_mask_diagnostics", sys.argv)
            return argparse.Namespace()
        def official(args):
            args.max_epoch, args.max_epoch_f = 2000, 300
            args.num_hidden, args.mask_rate = 1024, 0.5
            return args
        entry = types.SimpleNamespace(build_args=build, load_best_configs=official)
        for phase, epochs in (("smoke", 5), ("diagnostic", 100), ("development", 2000)):
            cli = argparse.Namespace(dataset="cora", device=0, seed=0, method="P", phase=phase)
            with mock.patch.object(sys, "argv", ["unchanged"]):
                args = recorded.official_arguments(entry, cli)
                self.assertEqual(sys.argv, ["unchanged"])
            self.assertEqual(args.max_epoch, epochs)
            self.assertEqual(args.num_hidden, 1024)
            self.assertEqual(args.mask_rate, 0.5)

    def test_feedback_stream_is_distinct_from_old_mask_statistics(self):
        masks, feedback = io.StringIO(), io.StringIO()
        recorder = recorded.RunRecorder(None, "cpu", masks, feedback)
        recorder.observe("stderr", "INFO INPUT_MASK_FEEDBACK {'step': 0, 'masked': 4}")
        self.assertEqual(recorder.feedback_records, 1)
        self.assertEqual(recorder.mask_records, 0)
        self.assertEqual(json.loads(feedback.getvalue())["masked"], 4)
        self.assertEqual(masks.getvalue(), "")

    def test_actual_source_contract_matches_and_legacy_recorder_is_frozen(self):
        root = Path(__file__).resolve().parent
        self.assertEqual(recorded.check_sources(root), recorded.CONTRACT)
        self.assertEqual(recorded.CONTRACT["record_full_batch.py"],
                         "7744e2e3ed17305b5b8428bc04f771c3a64eaa3e53b4c6331427c9055f025c95")

    def test_test_metrics_remain_hidden(self):
        text = "test_acc: 0.9999, TestAcc: 0.8888, Best ValAcc: 0.7000 in epoch 2 ---"
        redacted = recorded.redact(text)
        self.assertNotIn("0.9999", redacted)
        self.assertNotIn("0.8888", redacted)
        self.assertIn("0.7000", redacted)


HAS_RUNTIME = (importlib.util.find_spec("torch") is not None
               and importlib.util.find_spec("dgl") is not None)


@unittest.skipUnless(HAS_RUNTIME, "Requires actual PyTorch/DGL; must run on server")
class V2RecorderIntegrationTests(unittest.TestCase):
    def test_v2_recorder_preserves_training_and_saves_difficulty_artifacts(self):
        import numpy
        import torch
        import dgl
        import main_full_batch as entry
        import models.finetune as finetune
        from models import build_model
        import tempfile
        before_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        old_loader, old_builder, old_probe = entry.load_small_dataset, entry.build_model, entry.linear_probing_full_batch
        try:
            for device in [-1] + ([0] if torch.cuda.is_available() else []):
                for method in recorded.METHODS:
                    with self.subTest(device=device, method=method):
                        g = dgl.add_self_loop(dgl.to_bidirected(dgl.graph(
                            ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]), num_nodes=6)))
                        g.ndata["feat"] = torch.arange(48).float().reshape(6, 8) / 48
                        g.ndata["label"] = torch.tensor([0, 1, 0, 1, 0, 1])
                        for i, name in enumerate(("train_mask", "val_mask", "test_mask")):
                            ids = torch.arange(6)
                            g.ndata[name] = (ids >= i * 2) & (ids < i * 2 + 2)
                        entry.load_small_dataset = lambda name: (g, (8, 2))
                        cli = argparse.Namespace(dataset="cora", device=device, seed=0,
                                                 method=method, phase="smoke")
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            args = recorded.official_arguments(entry, cli)
                        args.num_hidden, args.num_heads = 16, 2
                        args.max_epoch, args.max_epoch_f = 3, 3
                        captured = {}
                        def builder(*a, **k):
                            captured["model"] = old_builder(*a, **k)
                            return captured["model"]
                        def probe(*a, **k):
                            result = old_probe(*a, **k)
                            if not k.get("mute", False):
                                captured["scores"] = result
                            return result
                        entry.build_model, entry.linear_probing_full_batch = builder, probe
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            entry.main(args)
                        expected_state = recorded.cpu_state(captured["model"], torch)
                        expected_rng = recorded.rng_digest(torch, numpy)
                        entry.build_model, entry.linear_probing_full_batch = old_builder, old_probe
                        stream = io.StringIO()
                        console = recorded.CaptureConsole(io.StringIO())
                        recorder = recorded.RunRecorder(torch,
                            torch.device("cpu" if device < 0 else "cuda:%d" % device),
                            feedback_stream=stream)
                        console.observer = recorder.observe
                        with console.redirect(), recorder.install(entry, finetune):
                            entry.main(args)
                        recorder.validate(args.max_epoch)
                        self.assertEqual(recorder.feedback_records, 3)
                        self.assertEqual(len(stream.getvalue().splitlines()), 3)
                        self.assertEqual(recorded.rng_digest(torch, numpy), expected_rng)
                        self.assertEqual(recorder.final_scores, tuple(float(x) for x in captured["scores"]))
                        for name, value in recorded.cpu_state(recorder.pretrained, torch).items():
                            torch.testing.assert_close(value, expected_state[name], rtol=1e-4, atol=1e-6)
                        with tempfile.TemporaryDirectory() as temporary:
                            path = Path(temporary)
                            recorded.save_models(path / "models.pt", recorder, args, torch, numpy)
                            sampler = recorder.pretrained.input_masker
                            sampler.export_node_statistics(path / "nodes.npz")
                            with numpy.load(path / "nodes.npz", allow_pickle=False) as nodes:
                                self.assertEqual(nodes["observation_count"].sum(), 9)
                            payload = torch.load(path / "models.pt", map_location="cpu", weights_only=True)
                            with contextlib.redirect_stdout(io.StringIO()):
                                fresh = build_model(argparse.Namespace(**payload["effective_args"]))
                            fresh.load_state_dict(payload["model_state_dict"], strict=True)
        finally:
            entry.load_small_dataset, entry.build_model, entry.linear_probing_full_batch = old_loader, old_builder, old_probe
            torch.set_num_threads(before_threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
