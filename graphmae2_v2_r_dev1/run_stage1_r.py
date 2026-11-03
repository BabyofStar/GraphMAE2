"""Run all regressions, then one R five-epoch GPU smoke check. No development batch."""

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
for pattern in ("test_input_masking.py", "test_masking_math_v2.py", "test_adaptive_masking_v2.py", "test_random_group_masking.py"):
    suite.addTests(loader.discover("tests", pattern=pattern))
suite.addTests(loader.loadTestsFromNames(["test_record_full_batch", "test_record_full_batch_r"]))
result = unittest.TextTestRunner(verbosity=2).run(suite)
print("R_DEV1_TEST_COUNT=" + str(result.testsRun))
print("R_DEV1_SKIPPED_COUNT=" + str(len(result.skipped)))
if result.testsRun != 80:
    print("Unexpected test count; expected 80")
if result.skipped:
    print("Server acceptance requires every CPU/CUDA integration test, with no skips.")
sys.exit(0 if result.wasSuccessful() and not result.skipped and result.testsRun == 80 else 1)
'''


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def run_command(command, repo, destination, lock_fd):
    print("RUN_LOG=" + str(destination), flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", DGLBACKEND="pytorch",
                       WANDB_MODE="disabled", MPLBACKEND="Agg", PYTHONUNBUFFERED="1", GRAPHMAE2_R_LOCK_FD=str(lock_fd))
    with destination.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=repo, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
    if result.returncode:
        print("\n".join(destination.read_text(encoding="utf-8", errors="replace").splitlines()[-70:]))
        raise RuntimeError("Command failed with exit %d: %s" % (result.returncode, destination.name))



def validate_r_smoke(run_dir):
    from fractions import Fraction
    import hashlib
    import math
    import numpy as np
    plan = json.loads((run_dir / 'random_group_plan.json').read_text(encoding='utf-8'))
    if plan['random_group_seed'] != 2026110301 or plan['nodes'] != 2708 or plan['masked'] != 1354:
        raise RuntimeError('R smoke grouping identity/budget mismatch')
    raw = [Fraction(plan['masked'] * size, plan['nodes']) for size in plan['group_sizes']]
    quotas = [math.floor(value) for value in raw]
    for i in sorted(range(len(raw)), key=lambda i: (-(raw[i] - quotas[i]), i))[:plan['masked'] - sum(quotas)]:
        quotas[i] += 1
    if plan['group_sizes'] != [677]*4 or quotas != plan['quotas']:
        raise RuntimeError('R smoke group sizes/quotas mismatch')
    records = [json.loads(line) for line in (run_dir / 'mask_feedback.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records) != 5:
        raise RuntimeError('R smoke must contain five feedback records')
    for row in records:
        if row['random_group_seed'] != plan['random_group_seed'] or row['sampling_group_order_sha256'] != plan['ordered_group_nodes_sha256']:
            raise RuntimeError('R smoke group fingerprint changed')
        if [x['nodes'] for x in row['sampling_groups']] != plan['group_sizes'] or [x['masked'] for x in row['sampling_groups']] != quotas:
            raise RuntimeError('R smoke actual sampled quotas differ')
        if sum(x['masked'] for x in row['density_layers']) != plan['masked']:
            raise RuntimeError('R smoke true density diagnostics missing')
    with np.load(run_dir / 'mask_node_statistics.npz', allow_pickle=False) as nodes:
        if set(nodes['node_id'].tolist()) != set(range(2708)) or int(nodes['observation_count'].sum()) != 5*1354:
            raise RuntimeError('R smoke node observation budget mismatch')
        order = nodes['sampling_group_node_order']
        if not np.array_equal(np.sort(order), np.arange(2708)):
            raise RuntimeError('R smoke group node order is not a permutation')
        if hashlib.sha256(order.astype('<i8', copy=False).tobytes()).hexdigest() != plan['ordered_group_nodes_sha256']:
            raise RuntimeError('R smoke saved group fingerprint differs')
        if np.bincount(nodes['sampling_group'], minlength=4).tolist() != plan['group_sizes']:
            raise RuntimeError('R smoke node group membership sizes differ')

def accept(repo, checkpoint_root):
    if Path(sys.prefix).resolve() != Path("/home/jovyan/work/envs/graphmae2").resolve():
        raise RuntimeError("Activate /home/jovyan/work/envs/graphmae2 before running")
    repo, checkpoint_root = repo.resolve(), checkpoint_root.resolve()
    from install_r import verify_files
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
        directory = Path(tempfile.mkdtemp(prefix="r_dev1_stage1_", dir=outputs))
        print("R_DEV1_STAGE1_DIR=" + str(directory), flush=True)
        status = {"status": "running", "stage": "implementation_acceptance_only",
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "reviewed_github_commit": manifest["reviewed_github_commit"],
                  "smoke_epochs": 5, "smoke_probe_epochs": 5,
                  "test_scores_used_for_selection": False}
        json_write(directory / "status.json", status)
        started = time.perf_counter()
        try:
            run_command([sys.executable, "-u", "-c", TEST_CODE], repo, directory / "tests.log", lock.fileno())
            print("R_DEV1_ALL_TESTS_PASSED; count=80; skipped=0", flush=True)
            for method in ("R",):
                run_dir = directory / (directory.name + "_cora_" + method + "_s0")
                run_command([sys.executable, "-u", "record_full_batch_r.py", "run",
                             "--dataset", "cora", "--method", method, "--seed", "0",
                             "--phase", "smoke", "--device", "0", "--output-dir", str(run_dir),
                             "--checkpoint-root", str(checkpoint_root)],
                            repo, directory / ("smoke_" + method + ".log"), lock.fileno())
                run_command([sys.executable, "-u", "record_full_batch_r.py", "verify",
                             "--output-dir", str(run_dir)], repo,
                            directory / ("reload_" + method + ".log"), lock.fileno())
                args = json.loads((run_dir / "effective_args.json").read_text(encoding="utf-8"))
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                if args["max_epoch"] != 5 or args["max_epoch_f"] != 5 or metrics["feedback_records"] != 5:
                    raise RuntimeError("Short smoke budget or feedback count changed")
                validate_r_smoke(run_dir)
                print("SMOKE_" + method + "=PASS (5 pretrain / 5 probe epochs)", flush=True)
            if verify_files(repo, manifest["after_sha256"]):
                raise RuntimeError("Source changed during acceptance")
            status.update(status="complete", wall_seconds=time.perf_counter() - started)
            json_write(directory / "status.json", status)
            print("R_DEV1_STAGE1_PASSED", flush=True)
            print("No performance conclusion: these are five-epoch engineering checks.")
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error))
            json_write(directory / "status.json", status)
            print("R_DEV1_STAGE1_FAILED; logs=" + str(directory), flush=True)
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
        print("R_DEV1_STAGE1_ERROR=" + str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
