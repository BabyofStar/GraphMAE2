"""Record the verified GraphMAE2/B3 entry point without replacing its training loop.

Use one process and one training seed per run. Test scores are written separately,
not printed. This saves final model weights, not a resumable optimizer checkpoint.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import types


VERSION = "1.0.0"
BASE_COMMIT = "bc473e8b750d6e90b7cba1c6bb20d858f617d90c"
METHODS = {
    "B0": "random",
    "B1": "structure_static",
    "B2": "structure_curriculum",
    "B3": "structure_curriculum_shuffled",
}
CONTRACT = {
    "main_full_batch.py": "10567a6173362ca3aca12218fce59d30597b8145ecbbddb4b604340ac9f7d0d2",
    "utils.py": "2df99927eac847e1880e1aa1d380bb9411c70b912cdbb753c95af64cd5c923f1",
    "models/__init__.py": "249ebfe22f6afa6e0f44f5987d2d66f35e5a9ed2e775bb44df867a385f72c7f7",
    "models/edcoder.py": "209584ec044e3053241ab3bb448d920bb0d2ba09a26c82d75a85d6d5ae9ca668",
    "models/input_masking.py": "230a45cf29255bfc46272f8774e5650aea46e85e48b558f82b286019e7d5d4b9",
    "models/finetune.py": "3c3f6e8f1154035a9a6d62420dc964f600cf672116c4fb3b12e7bb10f71de248",
    "models/gat.py": "2e536f955793d4190bfae9ed4161bedab53928ed43d2012bbb9e5e126d0a4a3c",
    "models/gcn.py": "55e3f0fac56ce78d7ed4856f83a3a84d61edde00be454f2db885b66ab992ac70",
    "models/loss_func.py": "b71a86e278b039ea77a2943b3ead804bb222ad28e437c379a1c22dd02e30140e",
    "datasets/data_proc.py": "d43cc47fb81e3d021dbaa15438b7a117054a5172855372f7fc4e17475671f88f",
    "datasets/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "datasets/lc_sampler.py": "0aca1c668e93bd6842220396a216a098b10e29aa9b3eb416ddd1fac472fea2cb",
    "datasets/localclustering.py": "a94c0b9f9c7db2656e75614aa66af6de2a2f8de82c0dab9609d7c490267aaf0c",
    "datasets/saint_sampler.py": "729cca3522d9927b65df00e8e6234ccf8f64140eddabe3332ee03c7eb4412a8e",
    "configs/cora.yaml": "3a5a6b87a73fd986ed0bd94ce224f37059739a7df7defa3cc89e6ad78f8aaca6",
    "configs/citeseer.yaml": "d9d9302aca6148a322861bb160300ff32e97e615f0638771ea1f5a2d9f3d7688",
    "configs/pubmed.yaml": "19f7897bab0097f2a6f3a85f9a7b02914cbb0e6627701832ea33c07aeb48a50a",
}
SUMMARY = re.compile(
    r"^--- TestAcc:.*?Best ValAcc:\s*([0-9.]+)\s+in epoch\s+(\d+)\s*---"
)


def redact(text):
    text = re.sub(
        r"(test_loss|test_acc|early-stopping-TestAcc|TestAcc):\s*[^,\s]+",
        r"\1:[HIDDEN]", text,
    )
    return re.sub(
        r"(# (?:final_acc|early-stopping_acc):)[^\r\n]*",
        r"\1 [HIDDEN]", text,
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    # All destinations belong to the newly created, empty run directory.
    path = Path(path)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    os.replace(temporary, path)


def check_sources(root):
    actual = {}
    for relative, expected in CONTRACT.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Missing or linked source: " + relative)
        actual[relative] = sha256(path)
        if actual[relative] != expected:
            raise RuntimeError("Source SHA256 mismatch; do not bypass: " + relative)
    return actual


def prepare_directories(output, checkpoint_root, repo_root):
    output = Path(output).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    repo_root = Path(repo_root).resolve()
    if checkpoint_root == repo_root or checkpoint_root.is_relative_to(repo_root):
        raise RuntimeError("Checkpoint root must be outside the Git project directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError("Output directory must be new or empty; never overwrite a run")
    output.mkdir(parents=True, exist_ok=True)
    weights = checkpoint_root / output.name
    weights.mkdir(parents=True, exist_ok=False)
    return output, weights


def snapshot_sources(root, output):
    files = {root / name for name in CONTRACT}
    files.add(Path(__file__).resolve())
    if (root / "test_record_full_batch.py").is_file():
        files.add(root / "test_record_full_batch.py")
    for directory in ("models", "datasets", "tests"):
        files.update(path for path in (root / directory).rglob("*.py")
                     if "__pycache__" not in path.parts)
    fingerprints = {}
    with tarfile.open(output / "source.tar", "x") as archive:
        for path in sorted(files):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise RuntimeError("Unexpected linked/outside source: " + str(path))
            name = path.relative_to(root).as_posix()
            fingerprints[name] = sha256(path)
            archive.add(path, arcname=name, recursive=False)
    write_json(output / "source_sha256.json", fingerprints)
    return fingerprints


def process_max_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


class CaptureStream(io.TextIOBase):
    """Buffer partial writes so split test-score messages cannot leak to disk."""

    def __init__(self, console, channel):
        self.console, self.channel, self.pending = console, channel, ""

    @property
    def encoding(self):
        return "utf-8"

    def isatty(self):
        return False

    def write(self, text):
        with self.console.lock:
            self.pending += str(text)
            parts = re.split(r"[\r\n]", self.pending)
            self.pending = parts.pop()
            for line in parts:
                if line:
                    self.console.emit(self.channel, line)
        return len(text)

    def flush(self):
        # Do not emit an incomplete record: the next write may contain its score.
        if not self.console.log.closed:
            self.console.log.flush()

    def finish(self):
        with self.console.lock:
            if self.pending:
                self.console.emit(self.channel, self.pending)
                self.pending = ""
            self.flush()


class CaptureConsole:
    def __init__(self, log):
        self.log, self.observer = log, None
        self.lock = threading.RLock()
        self.stdout = CaptureStream(self, "stdout")
        self.stderr = CaptureStream(self, "stderr")

    def emit(self, channel, line):
        if self.observer is not None:
            self.observer(channel, line)
        self.log.write(redact(line) + "\n")

    @staticmethod
    def handlers():
        loggers = [logging.getLogger()]
        loggers.extend(value for value in logging.Logger.manager.loggerDict.values()
                       if isinstance(value, logging.Logger))
        return {handler for logger in loggers for handler in logger.handlers}

    @contextlib.contextmanager
    def redirect(self):
        old_out, old_err = sys.stdout, sys.stderr
        changed = {}
        for handler in self.handlers():
            stream = getattr(handler, "stream", None)
            if stream is old_out or stream is old_err:
                changed[handler] = stream
                handler.setStream(self.stdout if stream is old_out else self.stderr)
        try:
            with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
                yield self
        finally:
            self.stdout.finish()
            self.stderr.finish()
            for handler in self.handlers():
                stream = getattr(handler, "stream", None)
                if handler in changed:
                    handler.setStream(changed[handler])
                elif stream is self.stdout or stream is self.stderr:
                    handler.setStream(old_out if stream is self.stdout else old_err)


class CopyCapture:
    """Delegate exactly one original deepcopy per original best-model update."""

    def __init__(self, original, observer):
        self.original, self.observer = original, observer

    def __getattr__(self, name):
        return getattr(self.original, name)

    def deepcopy(self, *args, **kwargs):
        result = self.original.deepcopy(*args, **kwargs)
        self.observer(result)
        return result


class RunRecorder:
    def __init__(self, torch, device, mask_stream=None):
        self.torch, self.device, self.mask_stream = torch, device, mask_stream
        self.pretrain_calls = self.intermediate_calls = self.final_calls = 0
        self.inner_calls = self.summary_count = 0
        self.pretrained = self.best_probe = self.last_probe = None
        self.final_scores = self.best_val_acc = self.best_val_epoch = None
        self.stage = "idle"
        self.resources, self.dataset = {}, None
        self.nonfinite = False
        self.mask_records = 0

    def sync(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def measured(self, name, function):
        self.sync()
        cuda = str(self.device).startswith("cuda")
        if cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)
        wall, cpu = time.perf_counter(), time.process_time()
        result = function()
        self.sync()
        self.resources[name] = {
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "pytorch_peak_allocated_bytes": (
                int(self.torch.cuda.max_memory_allocated(self.device)) if cuda else None),
            "pytorch_peak_reserved_bytes": (
                int(self.torch.cuda.max_memory_reserved(self.device)) if cuda else None),
            "process_lifetime_max_rss_bytes_at_phase_end": process_max_rss_bytes(),
        }
        return result

    def observe(self, channel, line):
        if re.search(r"\b(?:train_loss|val_loss):\s*(?:nan|[+-]?inf)\b", line, re.I):
            self.nonfinite = True
        if channel == "stdout" and self.stage == "final_probe":
            match = SUMMARY.search(line)
            if match:
                self.summary_count += 1
                self.best_val_acc, self.best_val_epoch = float(match[1]), int(match[2])
        mask = re.search(r"\bINPUT_MASK\s+(\{.*\})$", line)
        if mask:
            stats = ast.literal_eval(mask[1])
            if self.mask_stream is not None:
                self.mask_stream.write(json.dumps(stats, allow_nan=False) + "\n")
            self.mask_records += 1

    @contextlib.contextmanager
    def install(self, entry, finetune):
        train, probe, loader = entry.pretrain, entry.linear_probing_full_batch, entry.load_small_dataset

        def load_recorded(*args, **kwargs):
            result = self.measured("data_loading", lambda: loader(*args, **kwargs))
            graph, (features, classes) = result
            splits, split_hashes = {}, {}
            for name in ("train_mask", "val_mask", "test_mask"):
                value = graph.ndata[name].detach().cpu().contiguous()
                splits[name] = int(value.sum().item())
                split_hashes[name] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
            self.dataset = {
                "nodes": int(graph.num_nodes()), "edges_with_self_loops": int(graph.num_edges()),
                "features": int(features), "classes": int(classes),
                "splits": splits, "split_mask_sha256": split_hashes,
            }
            return result

        def train_recorded(*args, **kwargs):
            self.pretrain_calls += 1
            self.stage = "pretrain"
            try:
                result = self.measured("pretrain_including_intermediate_probes",
                                       lambda: train(*args, **kwargs))
                self.pretrained = result
                return result
            finally:
                self.stage = "idle"

        def probe_recorded(*args, **kwargs):
            bound = inspect.signature(probe).bind(*args, **kwargs)
            if bound.arguments.get("mute", False):
                self.intermediate_calls += 1
                return probe(*args, **kwargs)
            self.final_calls += 1
            self.stage = "final_probe"
            old_copy, old_inner = finetune.copy, finetune._linear_probing_full_batch

            def selected(model):
                self.best_probe = model

            def inner_recorded(*inner_args, **inner_kwargs):
                self.inner_calls += 1
                values = inspect.signature(old_inner).bind(*inner_args, **inner_kwargs)
                self.last_probe = values.arguments["model"]
                return old_inner(*inner_args, **inner_kwargs)

            finetune.copy = CopyCapture(old_copy, selected)
            finetune._linear_probing_full_batch = inner_recorded
            try:
                result = self.measured("final_embedding_and_linear_probe",
                                       lambda: probe(*args, **kwargs))
                self.final_scores = tuple(float(value) for value in result)
                return result
            finally:
                finetune.copy, finetune._linear_probing_full_batch = old_copy, old_inner
                self.stage = "idle"

        entry.pretrain, entry.linear_probing_full_batch, entry.load_small_dataset = (
            train_recorded, probe_recorded, load_recorded)
        try:
            yield self
        finally:
            entry.pretrain, entry.linear_probing_full_batch, entry.load_small_dataset = train, probe, loader

    def validate(self, epochs):
        if (self.pretrain_calls, self.final_calls, self.inner_calls, self.summary_count) != (1, 1, 1, 1):
            raise RuntimeError("Unexpected training/probe/summary call count")
        if self.intermediate_calls != epochs // 200:
            raise RuntimeError("Original intermediate evaluation frequency changed")
        if any(value is None for value in (self.pretrained, self.best_probe, self.last_probe, self.dataset)):
            raise RuntimeError("Missing model, classifier or dataset record")
        if self.nonfinite or not 0 <= self.best_val_acc <= 1:
            raise RuntimeError("Invalid loss/validation record")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.final_scores):
            raise RuntimeError("Invalid final evaluation result")


def rng_digest(torch, numpy):
    digest = hashlib.sha256(repr(random.getstate()).encode())
    state = numpy.random.get_state()
    digest.update(state[0].encode())
    digest.update(state[1].tobytes())
    digest.update(repr(state[2:]).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    for value in torch.cuda.get_rng_state_all():
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model, torch):
    state = model.state_dict()
    for key, value in list(state.items()):
        state[key] = value.detach().cpu().clone()
        if (state[key].is_floating_point() or state[key].is_complex()) and not torch.isfinite(state[key]).all().item():
            raise RuntimeError("Nonfinite model state: " + key)
    return state


def save_models(path, recorder, args, torch, numpy):
    before = rng_digest(torch, numpy)
    bundle = {
        "format_version": 1,
        "effective_args": vars(args).copy(),
        "model_state_dict": cpu_state(recorder.pretrained, torch),
        "linear_probe_best_state_dict": cpu_state(recorder.best_probe, torch),
        "linear_probe_last_state_dict": cpu_state(recorder.last_probe, torch),
        "probe_spec": {
            "in_features": int(recorder.last_probe.linear.in_features),
            "out_features": int(recorder.last_probe.linear.out_features),
        },
        "best_val_acc": recorder.best_val_acc,
        "best_val_epoch": recorder.best_val_epoch,
        "checkpoint_kind": "final_weights_not_training_resume",
    }
    with Path(path).open("xb") as stream:
        torch.save(bundle, stream)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    for group in ("model_state_dict", "linear_probe_best_state_dict", "linear_probe_last_state_dict"):
        if set(bundle[group]) != set(loaded[group]):
            raise RuntimeError("Saved checkpoint keys changed")
        for key, value in bundle[group].items():
            if not torch.equal(value, loaded[group][key]):
                raise RuntimeError("Saved checkpoint tensor changed: " + key)
    if before != rng_digest(torch, numpy):
        raise RuntimeError("Checkpoint recording changed global RNG state")
    return sha256(path)


def official_arguments(entry, cli):
    arguments = [
        "main_full_batch.py", "--dataset", cli.dataset, "--device", str(cli.device),
        "--use_cfg", "--seeds", str(cli.seed), "--input_mask_mode", METHODS[cli.method],
        "--structure_alpha_max", "0.5", "--structure_warmup_ratio", "0.2",
        "--structure_ramp_end_ratio", "0.8", "--structure_shuffle_seed", "20261025",
        "--mask_log_interval", "100",
    ]
    old = sys.argv
    try:
        sys.argv = arguments
        return entry.load_best_configs(entry.build_args())
    finally:
        sys.argv = old


def run(cli):
    started = time.perf_counter()
    root = Path.cwd().resolve()
    if Path(__file__).resolve().parent != root:
        raise RuntimeError("Upload this script into the repository root and run from that root")
    if cli.phase == "formal" and cli.seed not in range(10, 20):
        raise RuntimeError("Formal seeds are frozen to 10 through 19")
    if cli.seed < 0:
        raise RuntimeError("Seed must be nonnegative")
    core = check_sources(root)
    output, weights = prepare_directories(cli.output_dir, cli.checkpoint_root, root)
    checkpoint = weights / "models.pt"
    status = {"status": "running", "phase": cli.phase, "method": cli.method,
              "dataset": cli.dataset, "seed": cli.seed, "recorder_version": VERSION}
    write_json(output / "status.json", status)
    print("RECORD_DIR=" + str(output), flush=True)
    print("CHECKPOINT_DIR=" + str(weights), flush=True)
    with (output / "train.log").open("x", encoding="utf-8", buffering=1) as log, \
         (output / "mask_stats.jsonl").open("x", encoding="utf-8", buffering=1) as masks:
        console = CaptureConsole(log)
        with console.redirect():
            try:
                snapshots = snapshot_sources(root, output)
                import torch
                import numpy
                import dgl
                import main_full_batch as entry
                import models.finetune as finetune
                versions = {"torch": str(torch.__version__), "dgl": str(dgl.__version__),
                            "numpy": str(numpy.__version__)}
                if versions != {"torch": "2.1.2+cu118", "dgl": "1.1.3+cu118", "numpy": "1.26.4"}:
                    raise RuntimeError("Training package versions differ from the verified baseline: " + str(versions))
                if not torch.cuda.is_available() or not 0 <= cli.device < torch.cuda.device_count():
                    raise RuntimeError("Requested CUDA device is unavailable")
                device = torch.device("cuda", cli.device)
                args = official_arguments(entry, cli)
                print(args)
                write_json(output / "effective_args.json", vars(args))
                metadata = {
                    "recorder_version": VERSION, "python": sys.version,
                    "platform": platform.platform(), "versions": versions,
                    "torch_cuda_runtime": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(device),
                    "visible_gpu_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
                    "torch_cpu_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "runtime_environment": {
                        name: os.environ.get(name) for name in (
                            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
                            "PYTHONHASHSEED", "DGLBACKEND")
                    },
                    "core_sha256": core,
                    "packages": sorted((dist.metadata.get("Name") or "", dist.version)
                                       for dist in importlib.metadata.distributions()),
                }
                head = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                      capture_output=True, text=True).stdout.strip()
                metadata["git_head"] = head
                subprocess.run(["git", "cat-file", "-e", BASE_COMMIT + "^{commit}"], check=True)
                write_json(output / "environment.json", metadata)
                recorder = RunRecorder(torch, device, masks)
                console.observer = recorder.observe
                with recorder.install(entry, finetune):
                    entry.main(args)
                recorder.validate(args.max_epoch)
                # main() sets num_features during loading; export the final effective namespace.
                write_json(output / "effective_args.json", vars(args))
                if any(sha256(root / name) != value for name, value in snapshots.items()):
                    raise RuntimeError("Source changed during the run")
                save_start = time.perf_counter()
                checkpoint_hash = save_models(checkpoint, recorder, args, torch, numpy)
                resources = recorder.resources
                resources["checkpoint_write_and_readback_wall_seconds"] = time.perf_counter() - save_start
                resources["runner_wall_seconds_to_artifact_export"] = time.perf_counter() - started
                write_json(output / "resources.json", resources)
                write_json(output / "test_metrics.json", {
                    "accuracy_scale": "fraction", "primary": "val_selected_test_acc",
                    "final_test_acc": recorder.final_scores[0],
                    "val_selected_test_acc": recorder.final_scores[1],
                    "selection": "final_pretrained_encoder_then_validation_selected_linear_classifier",
                })
                metrics = dict(status, status="complete", best_val_acc=recorder.best_val_acc,
                               best_val_epoch=recorder.best_val_epoch, dataset_info=recorder.dataset,
                               intermediate_probe_calls=recorder.intermediate_calls,
                               mask_records=recorder.mask_records, checkpoint_path=str(checkpoint),
                               checkpoint_sha256=checkpoint_hash, checkpoint_readback_equal=True,
                               test_metrics_file="test_metrics.json")
                write_json(output / "metrics.json", metrics)
                write_json(output / "status.json", dict(status, status="complete"))
            except BaseException as error:
                traceback.print_exc()
                write_json(output / "status.json", dict(status, status="failed",
                           error_type=type(error).__name__, error=redact(str(error))))
                raise
    print("Best ValAcc: %.4f in epoch %d" % (recorder.best_val_acc, recorder.best_val_epoch))
    print("Pretrain wall seconds: %.3f" % resources["pretrain_including_intermediate_probes"]["wall_seconds"])
    print("Final probe wall seconds: %.3f" % resources["final_embedding_and_linear_probe"]["wall_seconds"])
    print("CHECKPOINT_READBACK=PASS")
    print("RECORDED_RUN=PASS")


def verify(cli):
    output = Path(cli.output_dir).resolve()
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise RuntimeError("Only a complete run can be verified")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    path = Path(metrics["checkpoint_path"])
    if sha256(path) != metrics["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA256 mismatch")
    check_sources(Path.cwd().resolve())
    import torch
    from models import build_model
    from models.finetune import LogisticRegression
    payload = torch.load(path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(**payload["effective_args"])
    model = build_model(args)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for name in ("linear_probe_best_state_dict", "linear_probe_last_state_dict"):
        probe = LogisticRegression(payload["probe_spec"]["in_features"],
                                   payload["probe_spec"]["out_features"])
        probe.load_state_dict(payload[name], strict=True)
    print("CHECKPOINT_SHA256=PASS")
    print("MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS")
    print("This is a weight-loading check, not resumed training or a new accuracy evaluation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--dataset", required=True, choices=("cora", "citeseer", "pubmed"))
    run_parser.add_argument("--method", required=True, choices=tuple(METHODS))
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--phase", required=True, choices=("smoke", "formal"))
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--checkpoint-root", required=True, type=Path)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--output-dir", required=True, type=Path)
    cli = parser.parse_args()
    try:
        run(cli) if cli.command == "run" else verify(cli)
    except Exception as error:
        print("RECORDER_ERROR: " + redact(str(error)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
