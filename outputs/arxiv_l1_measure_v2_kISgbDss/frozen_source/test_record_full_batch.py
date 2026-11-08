"""Run beside record_full_batch.py: python -m unittest -v test_record_full_batch.

Standard-library tests run locally. Real-model CPU/CUDA tests additionally need
the GraphMAE2 repository, PyTorch and DGL; no actual dataset is downloaded.
"""

import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import types
import unittest

import record_full_batch as recorded


class RecorderUtilityTests(unittest.TestCase):
    def test_all_test_score_formats_are_redacted(self):
        text = ("test_loss: 0.7123, test_acc: 0.8123\n"
                "--- TestAcc: 0.8123, early-stopping-TestAcc: 0.8234, "
                "Best ValAcc: 0.8040 in epoch 71 ---\n"
                "# final_acc: 0.8123+/-0.0123\n"
                "# early-stopping_acc: 0.8234+/-0.0123\n")
        result = recorded.redact(text)
        self.assertNotIn("0.8123", result)
        self.assertNotIn("0.8234", result)
        self.assertNotIn("0.7123", result)
        self.assertNotIn("0.0123", result)
        self.assertIn("Best ValAcc: 0.8040 in epoch 71", result)

    def test_split_writes_and_flush_never_emit_partial_test_scores(self):
        sink = io.StringIO()
        console = recorded.CaptureConsole(sink)
        console.stdout.write("--- Test")
        console.stdout.flush()
        self.assertEqual(sink.getvalue(), "")
        console.stdout.write("Acc: 0.8123, Best ValAcc: 0.8040 in epoch 71 ---")
        console.stdout.flush()
        self.assertEqual(sink.getvalue(), "")
        console.stdout.write("\n")
        self.assertNotIn("0.8123", sink.getvalue())
        self.assertIn("Best ValAcc: 0.8040", sink.getvalue())

    def test_carriage_returns_and_observer_preserve_record_boundaries(self):
        sink, observed = io.StringIO(), []
        console = recorded.CaptureConsole(sink)
        console.observer = lambda channel, line: observed.append((channel, line))
        console.stderr.write("\rtrain_loss: 0.4\rtest_acc: 0.8123\r")
        self.assertEqual(len(observed), 2)
        self.assertNotIn("0.8123", sink.getvalue())
        self.assertIn("train_loss: 0.4", sink.getvalue())

    def test_final_validation_summary_is_distinct_from_ignore_probe(self):
        recorder = recorded.RunRecorder(None, "cpu")
        recorder.stage = "pretrain"
        recorder.observe("stdout", "# IGNORE: --- TestAcc: 0.7, Best ValAcc: 0.9 in epoch 2 ---")
        self.assertEqual(recorder.summary_count, 0)
        recorder.stage = "final_probe"
        recorder.observe("stdout", "--- TestAcc: 0.7, Best ValAcc: 0.8040 in epoch 71 ---")
        self.assertEqual(recorder.summary_count, 1)
        self.assertEqual(recorder.best_val_acc, .804)
        self.assertEqual(recorder.best_val_epoch, 71)

    def test_nonfinite_training_loss_is_flagged(self):
        for value in ("nan", "inf", "-inf"):
            recorder = recorded.RunRecorder(None, "cpu")
            recorder.observe("stderr", "# Epoch 1, train_loss: " + value)
            self.assertTrue(recorder.nonfinite)

    def test_mask_diagnostics_are_exported_without_test_scores(self):
        sink = io.StringIO()
        recorder = recorded.RunRecorder(None, "cpu", sink)
        recorder.observe("stderr", "INFO - INPUT_MASK {'step': 0, 'alpha': 0.0, 'masked': 2}")
        self.assertEqual(json.loads(sink.getvalue())["masked"], 2)
        self.assertEqual(recorder.mask_records, 1)

    def test_copy_proxy_delegates_once_and_keeps_the_same_returned_object(self):
        calls, captured = [], []
        def deepcopy(value):
            calls.append(value)
            return {"copy": value}
        original = types.SimpleNamespace(deepcopy=deepcopy, other="unchanged")
        proxy = recorded.CopyCapture(original, captured.append)
        result = proxy.deepcopy(7)
        self.assertEqual(calls, [7])
        self.assertIs(result, captured[0])
        self.assertEqual(proxy.other, "unchanged")

    def test_install_restores_original_functions_after_exception(self):
        def function(*args, **kwargs):
            raise RuntimeError("expected failure")
        entry = types.SimpleNamespace(pretrain=function, linear_probing_full_batch=function,
                                      load_small_dataset=function)
        finetune = types.SimpleNamespace(copy=copy, _linear_probing_full_batch=function)
        recorder = recorded.RunRecorder(None, "cpu")
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            with recorder.install(entry, finetune):
                entry.pretrain()
        self.assertIs(entry.pretrain, function)
        self.assertIs(entry.linear_probing_full_batch, function)
        self.assertIs(entry.load_small_dataset, function)
        self.assertIs(finetune.copy, copy)
        self.assertIs(finetune._linear_probing_full_batch, function)

    def test_muted_probe_is_delegated_without_final_probe_capture(self):
        calls = []
        def probe(value, mute=False):
            calls.append((value, mute))
            return (.7, .8)
        function = lambda *args, **kwargs: None
        entry = types.SimpleNamespace(pretrain=function, linear_probing_full_batch=probe,
                                      load_small_dataset=function)
        finetune = types.SimpleNamespace(copy=copy, _linear_probing_full_batch=function)
        recorder = recorded.RunRecorder(None, "cpu")
        with recorder.install(entry, finetune):
            self.assertEqual(entry.linear_probing_full_batch(7, mute=True), (.7, .8))
        self.assertEqual(calls, [(7, True)])
        self.assertEqual(recorder.intermediate_calls, 1)
        self.assertEqual(recorder.final_calls, 0)

    def test_final_probe_failure_restores_copy_and_inner_function(self):
        def function(*args, **kwargs):
            return None
        def failing_probe(value, mute=False):
            raise RuntimeError("expected probe failure")
        entry = types.SimpleNamespace(pretrain=function, linear_probing_full_batch=failing_probe,
                                      load_small_dataset=function)
        finetune = types.SimpleNamespace(copy=copy, _linear_probing_full_batch=function)
        recorder = recorded.RunRecorder(None, "cpu")
        with self.assertRaisesRegex(RuntimeError, "expected probe failure"):
            with recorder.install(entry, finetune):
                entry.linear_probing_full_batch(7)
        self.assertIs(finetune.copy, copy)
        self.assertIs(finetune._linear_probing_full_batch, function)
        self.assertIs(entry.linear_probing_full_batch, failing_probe)
        self.assertEqual(recorder.stage, "idle")

    def test_official_arguments_freeze_protocol_and_restore_sys_argv(self):
        import sys
        old_argv = sys.argv
        entry = types.SimpleNamespace(
            build_args=lambda: tuple(sys.argv), load_best_configs=lambda value: value)
        for method, mode in recorded.METHODS.items():
            cli = argparse.Namespace(dataset="pubmed", device=0, seed=17, method=method)
            argv = recorded.official_arguments(entry, cli)
            self.assertIn("--use_cfg", argv)
            self.assertEqual(argv[argv.index("--input_mask_mode") + 1], mode)
            self.assertEqual(argv[argv.index("--seeds") + 1], "17")
            self.assertEqual(argv[argv.index("--structure_alpha_max") + 1], "0.5")
            self.assertEqual(argv[argv.index("--structure_shuffle_seed") + 1], "20261025")
            self.assertIs(sys.argv, old_argv)

    def test_nonempty_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="graphmae2_record_tests_") as temporary:
            root = Path(temporary)
            project, output, checkpoints = root / "project", root / "output", root / "weights"
            project.mkdir()
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.touch()
            with self.assertRaisesRegex(RuntimeError, "never overwrite"):
                recorded.prepare_directories(output, checkpoints, project)
            self.assertTrue(sentinel.exists())

    def test_checkpoint_directory_inside_project_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="graphmae2_record_tests_") as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "outside"):
                recorded.prepare_directories(root / "run", root / "weights", root)

    def test_json_rejects_nonfinite_values_and_source_contract_has_expected_base(self):
        with tempfile.TemporaryDirectory(prefix="graphmae2_record_tests_") as temporary:
            with self.assertRaises(ValueError):
                recorded.write_json(Path(temporary) / "bad.json", {"value": float("nan")})
        self.assertEqual(recorded.CONTRACT["main_full_batch.py"],
                         "10567a6173362ca3aca12218fce59d30597b8145ecbbddb4b604340ac9f7d0d2")


HAS_RUNTIME = (importlib.util.find_spec("torch") is not None
               and importlib.util.find_spec("dgl") is not None
               and importlib.util.find_spec("main_full_batch") is not None)


@unittest.skipUnless(HAS_RUNTIME, "Requires GraphMAE2 repository, PyTorch and DGL")
class ActualModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import dgl
        import numpy
        import main_full_batch
        import models.finetune
        cls.torch, cls.dgl, cls.numpy = torch, dgl, numpy
        cls.entry, cls.finetune = main_full_batch, models.finetune
        cls.old_threads = torch.get_num_threads()
        # Test-only: keep tiny-graph CPU regressions fast; restore on class exit.
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def arguments(self, mode, device):
        cli = argparse.Namespace(dataset="cora", device=device, seed=0, method=mode)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            args = recorded.official_arguments(self.entry, cli)
        args.num_hidden, args.num_heads = 16, 2
        args.max_epoch, args.max_epoch_f = 3, 3
        args.mask_log_interval = 0
        return args

    def graph(self):
        torch, dgl = self.torch, self.dgl
        torch.manual_seed(1234)
        # Fixture-only fix v1.0.1: official accuracy() squeezes labels before
        # len(), so every split needs at least two nodes, not a singleton.
        graph = dgl.add_self_loop(dgl.graph(
            ([0, 1, 2, 3, 4, 5, 0, 2], [1, 2, 3, 4, 5, 0, 2, 0]),
            num_nodes=6))
        graph.ndata["feat"] = torch.randn(6, 8)
        graph.ndata["label"] = torch.tensor([0, 1, 0, 1, 0, 1])
        graph.ndata["train_mask"] = torch.tensor([True, True, False, False, False, False])
        graph.ndata["val_mask"] = torch.tensor([False, False, True, True, False, False])
        graph.ndata["test_mask"] = torch.tensor([False, False, False, False, True, True])
        for split in ("train_mask", "val_mask", "test_mask"):
            self.assertEqual(int(graph.ndata[split].sum().item()), 2)
            self.assertEqual(graph.ndata["label"][graph.ndata[split]].squeeze().ndim, 1)
        return graph

    def original_run(self, graph, args):
        entry = self.entry
        loader, builder, probe = entry.load_small_dataset, entry.build_model, entry.linear_probing_full_batch
        captured = {}
        def build(*a, **k):
            result = builder(*a, **k)
            captured["model"] = result
            return result
        def evaluate(*a, **k):
            result = probe(*a, **k)
            if not k.get("mute", False):
                captured["scores"] = result
            return result
        entry.load_small_dataset = lambda name: (graph, (8, 2))
        entry.build_model, entry.linear_probing_full_batch = build, evaluate
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                entry.main(args)
            captured["rng"] = recorded.rng_digest(self.torch, self.numpy)
            captured["state"] = recorded.cpu_state(captured["model"], self.torch)
            return captured
        finally:
            entry.load_small_dataset, entry.build_model, entry.linear_probing_full_batch = loader, builder, probe

    def recorded_run(self, graph, args):
        entry, loader = self.entry, self.entry.load_small_dataset
        entry.load_small_dataset = lambda name: (graph, (8, 2))
        sink = io.StringIO()
        console = recorded.CaptureConsole(sink)
        device = self.torch.device("cpu" if args.device < 0 else "cuda:%d" % args.device)
        recorder = recorded.RunRecorder(self.torch, device)
        console.observer = recorder.observe
        try:
            with console.redirect(), recorder.install(entry, self.finetune):
                entry.main(args)
            recorder.validate(args.max_epoch)
            return recorder
        finally:
            entry.load_small_dataset = loader

    def test_recording_preserves_real_training_results_weights_and_rng(self):
        devices = [-1] + ([0] if self.torch.cuda.is_available() else [])
        for device in devices:
            for method in ("B0", "B1", "B2", "B3"):
                with self.subTest(device=device, method=method):
                    graph = self.graph()
                    old = self.original_run(graph, self.arguments(method, device))
                    new = self.recorded_run(graph, self.arguments(method, device))
                    self.assertEqual(old["rng"], recorded.rng_digest(self.torch, self.numpy))
                    new_state = recorded.cpu_state(new.pretrained, self.torch)
                    for key, value in old["state"].items():
                        self.torch.testing.assert_close(value, new_state[key], rtol=1e-4, atol=1e-6)
                    self.assertEqual(tuple(float(x) for x in old["scores"]), new.final_scores)

    def test_actual_checkpoint_cpu_reload_and_save_rng_invariance(self):
        from models import build_model
        from models.finetune import LogisticRegression
        args = self.arguments("B3", 0 if self.torch.cuda.is_available() else -1)
        recorder = self.recorded_run(self.graph(), args)
        with tempfile.TemporaryDirectory(prefix="graphmae2_record_tests_") as temporary:
            checkpoint = Path(temporary) / "models.pt"
            before = recorded.rng_digest(self.torch, self.numpy)
            digest = recorded.save_models(checkpoint, recorder, args, self.torch, self.numpy)
            self.assertEqual(before, recorded.rng_digest(self.torch, self.numpy))
            self.assertEqual(digest, recorded.sha256(checkpoint))
            payload = self.torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = build_model(argparse.Namespace(**payload["effective_args"]))
            model.load_state_dict(payload["model_state_dict"], strict=True)
            for key in ("linear_probe_best_state_dict", "linear_probe_last_state_dict"):
                probe = LogisticRegression(16, 2)
                probe.load_state_dict(payload[key], strict=True)
            with self.assertRaises(FileExistsError):
                recorded.save_models(checkpoint, recorder, args, self.torch, self.numpy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
