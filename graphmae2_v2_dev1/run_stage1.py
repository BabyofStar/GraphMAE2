"""Run all regressions, then two five-epoch GPU smoke checks. No development batch."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True


TEST_CODE = '''
import sys
import unittest
loader = unittest.TestLoader()
suite = unittest.TestSuite()
for pattern in ("test_input_masking.py", "test_masking_math_v2.py", "test_adaptive_masking_v2.py"):
    suite.addTests(loader.discover("tests", pattern=pattern))
suite.addTests(loader.loadTestsFromNames(["test_record_full_batch", "test_record_full_batch_v2"]))
result = unittest.TextTestRunner(verbosity=2).run(suite)
print("V2_TEST_COUNT=" + str(result.testsRun))
print("V2_SKIPPED_COUNT=" + str(len(result.skipped)))
if result.skipped:
    print("Server acceptance requires every CPU/CUDA integration test, with no skips.")
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
'''


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def run_command(command, repo, destination):
    print("RUN_LOG=" + str(destination), flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", DGLBACKEND="pytorch",
                       WANDB_MODE="disabled", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    with destination.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=repo, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(destination.read_text(encoding="utf-8", errors="replace").splitlines()[-70:]))
        raise RuntimeError("Command failed with exit %d: %s" % (result.returncode, destination.name))


def accept(repo, checkpoint_root):
    if Path(sys.prefix).resolve() != Path("/home/jovyan/work/envs/graphmae2").resolve():
        raise RuntimeError("Activate /home/jovyan/work/envs/graphmae2 before running")
    repo, checkpoint_root = repo.resolve(), checkpoint_root.resolve()
    from install_v2 import verify_files
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text(encoding="utf-8"))
    failures = verify_files(repo, manifest["after_sha256"])
    if failures:
        raise RuntimeError("Install/source verification required: " + ", ".join(failures))
    import torch
    import dgl
    import numpy
    versions = (str(torch.__version__), str(dgl.__version__), str(numpy.__version__))
    if versions != ("2.1.2+cu118", "1.1.3+cu118", "1.26.4"):
        raise RuntimeError("Unexpected runtime versions: " + str(versions))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU-only acceptance is insufficient")
    print("PYTHON=" + sys.executable)
    print("GPU=" + torch.cuda.get_device_name(0))
    print("VISIBLE_GPU_GIB=%.3f" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    subprocess.run(["git", "cat-file", "-e",
                    "bc473e8b750d6e90b7cba1c6bb20d858f617d90c^{commit}"], cwd=repo, check=True)
    outputs = repo / "outputs"
    outputs.mkdir(exist_ok=True)
    # Share the existing formal-run lock: an old batch and these GPU checks must not overlap.
    import fcntl
    with (outputs / ".formal_record_v1.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another recorded experiment holds the GPU batch lock")
        directory = Path(tempfile.mkdtemp(prefix="v2_stage1_", dir=outputs))
        print("V2_STAGE1_DIR=" + str(directory), flush=True)
        status = {"status": "running", "stage": "implementation_acceptance_only",
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "reviewed_github_commit": manifest["reviewed_github_commit"],
                  "smoke_epochs": 5, "smoke_probe_epochs": 5,
                  "test_scores_used_for_selection": False}
        json_write(directory / "status.json", status)
        started = time.perf_counter()
        try:
            run_command([sys.executable, "-u", "-c", TEST_CODE], repo, directory / "tests.log")
            print("V2_ALL_TESTS_PASSED; skipped=0", flush=True)
            for method in ("D", "S"):
                run_dir = directory / (directory.name + "_cora_" + method + "_s0")
                run_command([sys.executable, "-u", "record_full_batch_v2.py", "run",
                             "--dataset", "cora", "--method", method, "--seed", "0",
                             "--phase", "smoke", "--device", "0", "--output-dir", str(run_dir),
                             "--checkpoint-root", str(checkpoint_root)],
                            repo, directory / ("smoke_" + method + ".log"))
                run_command([sys.executable, "-u", "record_full_batch_v2.py", "verify",
                             "--output-dir", str(run_dir)], repo,
                            directory / ("reload_" + method + ".log"))
                args = json.loads((run_dir / "effective_args.json").read_text(encoding="utf-8"))
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                if args["max_epoch"] != 5 or args["max_epoch_f"] != 5 or metrics["feedback_records"] != 5:
                    raise RuntimeError("Short smoke budget or feedback count changed")
                print("SMOKE_" + method + "=PASS (5 pretrain / 5 probe epochs)", flush=True)
            if verify_files(repo, manifest["after_sha256"]):
                raise RuntimeError("Source changed during acceptance")
            status.update(status="complete", wall_seconds=time.perf_counter() - started)
            json_write(directory / "status.json", status)
            print("V2_STAGE1_PASSED", flush=True)
            print("No performance conclusion: these are five-epoch engineering checks.")
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error))
            json_write(directory / "status.json", status)
            print("V2_STAGE1_FAILED; logs=" + str(directory), flush=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("/home/jovyan/work/GraphMAE2_artifacts/checkpoints"))
    args = parser.parse_args()
    try:
        accept(args.repo, args.checkpoint_root)
    except Exception as error:
        print("V2_STAGE1_ERROR=" + str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
