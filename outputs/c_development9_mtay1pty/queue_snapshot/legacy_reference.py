"""Run only the 18 full-budget B0/B2 references after v2 stage-1 acceptance."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True

MANIFEST_SHA256 = "18243c6b74c11202bd66097a84877adfba8b0c36462b45630480cfc3f74c2c64"
RECORDER_SHA256 = "033012c493b602357e4fe55d3758c91586593b6b27210f5cf9dcfd9592ef8fc0"
METHODS = {"B0": "random", "B2": "structure_curriculum"}
CONFIGS = {
    "cora": {"max_epoch": 2000, "max_epoch_f": 300, "num_hidden": 1024,
             "num_heads": 8, "mask_rate": 0.5, "lr": 0.001, "lr_f": 0.025,
             "weight_decay": 0.0002, "weight_decay_f": 0.0001,
             "alpha_l": 3, "lam": 0.15, "momentum": 0, "replace_rate": 0.15},
    "citeseer": {"max_epoch": 500, "max_epoch_f": 500, "num_hidden": 1024,
                 "num_heads": 4, "mask_rate": 0.5, "lr": 0.0005, "lr_f": 0.025,
                 "weight_decay": 0.0001, "weight_decay_f": 0.01,
                 "alpha_l": 1, "lam": 0.1, "momentum": 1, "replace_rate": 0.0},
    "pubmed": {"max_epoch": 2000, "max_epoch_f": 500, "num_hidden": 512,
               "num_heads": 2, "mask_rate": 0.9, "lr": 0.005, "lr_f": 0.025,
               "weight_decay": 0.00001, "weight_decay_f": 0.0005,
               "alpha_l": 4, "lam": 1, "momentum": 0.995, "replace_rate": 0.0},
}
COMMON = {"remask_method": "fixed", "use_cfg": True, "linear_prob": True,
          "scheduler": True, "input_mask_diagnostics": True,
          "structure_alpha_max": 0.5, "structure_warmup_ratio": 0.2,
          "structure_ramp_end_ratio": 0.8, "difficulty_ema_beta": 0.9,
          "difficulty_clip_quantile": 0.95, "density_bins": 4,
          "density_chunk_size": 256, "mask_log_interval": 100,
          "encoder": "gat", "decoder": "gat", "num_layers": 2,
          "num_dec_layers": 1, "num_out_heads": 1, "loss_fn": "sce"}
DATA = {
    "cora": {"nodes": 2708, "edges_with_self_loops": 13264, "features": 1433,
             "classes": 7, "splits": {"train_mask": 140, "val_mask": 500, "test_mask": 1000}},
    "citeseer": {"nodes": 3327, "edges_with_self_loops": 12431, "features": 3703,
                 "classes": 6, "splits": {"train_mask": 120, "val_mask": 500, "test_mask": 1000}},
    "pubmed": {"nodes": 19717, "edges_with_self_loops": 108365, "features": 500,
               "classes": 3, "splits": {"train_mask": 60, "val_mask": 500, "test_mask": 1000}},
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def combinations():
    # Paired methods run adjacently; every child process resets its training seed.
    return [(dataset, method, seed) for dataset in CONFIGS
            for seed in (0, 1, 2) for method in METHODS]


def check_sources(repo):
    manifest_path = repo / "graphmae2_v2_dev1/manifest.json"
    require(sha256(manifest_path) == MANIFEST_SHA256, "v2-dev1 manifest differs")
    manifest = read_json(manifest_path)
    for name, expected in manifest["after_sha256"].items():
        path = repo / name
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(repo) and sha256(path) == expected,
                "Source differs from accepted v2-dev1: " + name)
    require(sha256(repo / "record_full_batch_v2.py") == RECORDER_SHA256,
            "v2 recorder differs from accepted version")
    return manifest


def check_stage1(repo, directory):
    directory = directory.resolve()
    require(directory.is_relative_to(repo / "outputs"), "Stage-1 directory must be in project outputs")
    status = read_json(directory / "status.json")
    require(status.get("status") == "complete"
            and status.get("stage") == "implementation_acceptance_only",
            "Stage-1 acceptance is not complete")
    log = (directory / "tests.log").read_text(encoding="utf-8", errors="replace")
    require("V2_TEST_COUNT=72" in log and "V2_SKIPPED_COUNT=0" in log,
            "Stage-1 test-count/no-skip evidence is missing")
    for method in ("D", "S"):
        path = directory / (directory.name + "_cora_" + method + "_s0")
        metrics = read_json(path / "metrics.json")
        args = read_json(path / "effective_args.json")
        require(metrics.get("status") == "complete" and metrics.get("phase") == "smoke"
                and metrics.get("method") == method and args.get("max_epoch") == 5
                and args.get("max_epoch_f") == 5, "Short-smoke evidence differs: " + method)
        reload_log = (directory / ("reload_" + method + ".log")).read_text(encoding="utf-8")
        require("MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS" in reload_log,
                "Stage-1 CPU-reload evidence is missing: " + method)


def validate_run(run_dir, dataset, method, seed):
    status, metrics, args, resources = [read_json(run_dir / name) for name in
        ("status.json", "metrics.json", "effective_args.json", "resources.json")]
    for record in (status, metrics):
        for key, expected in {"status": "complete", "phase": "development",
                              "dataset": dataset, "method": method, "seed": seed,
                              "official_epoch_budget": True}.items():
            require(record.get(key) == expected, "Run metadata differs: " + key)
    for key, expected in dict(COMMON, **CONFIGS[dataset], dataset=dataset,
                              seeds=[seed], input_mask_mode=METHODS[method]).items():
        require(args.get(key) == expected, "Effective configuration differs: " + key)
    info = metrics["dataset_info"]
    for key, expected in DATA[dataset].items():
        require(info.get(key) == expected, "Dataset/split statistics differ: " + key)
    for key in ("feature_matrix_sha256", "ordered_edge_index_sha256"):
        require(isinstance(info.get(key), str) and len(info[key]) == 64,
                "Dataset fingerprint is missing: " + key)
    require(set(info["split_mask_sha256"]) == {"train_mask", "val_mask", "test_mask"},
            "Split fingerprints are incomplete")
    val = metrics["best_val_acc"]
    require(isinstance(val, (int, float)) and math.isfinite(val) and 0 <= val <= 1,
            "Invalid validation accuracy")
    epochs = CONFIGS[dataset]["max_epoch"]
    expected_steps = sorted(set(range(0, epochs, 100)) | {epochs - 1})
    feedback = [json.loads(line) for line in (run_dir / "mask_feedback.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    require([item["step"] for item in feedback] == expected_steps,
            "Feedback steps differ from the complete official epoch budget")
    require(metrics["feedback_records"] == len(expected_steps), "Feedback count differs")
    n, k = info["nodes"], int(info["nodes"] * args["mask_rate"])
    for item in feedback:
        require(item["mode"] == METHODS[method] and item["masked"] == k
                and item["nodes"] == n, "Mask mode/node budget differs")
        progress = item["step"] / max(epochs - 1, 1)
        expected_alpha = (0.0 if method == "B0" else
                          0.5 * min(max((progress - 0.2) / (0.8 - 0.2), 0.0), 1.0))
        require(math.isclose(item["alpha"], expected_alpha, abs_tol=1e-12),
                "Degree-curriculum alpha schedule differs")
        mean_count = item["observation_count_min_mean_max"][1]
        require(math.isclose(mean_count, k * (item["step"] + 1) / n,
                             rel_tol=1e-9, abs_tol=1e-8), "Observation budget differs")
    node_path = run_dir / metrics["mask_node_statistics_file"]
    require(sha256(node_path) == metrics["mask_node_statistics_sha256"],
            "Node-statistic checksum differs")
    seconds = resources["pretrain_including_intermediate_probes"]["wall_seconds"]
    require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0,
            "Invalid resource record")
    # Deliberately never open test_metrics.json during development selection.
    return {"dataset": dataset, "method": method, "seed": seed,
            "best_val_acc": val, "best_val_epoch": metrics["best_val_epoch"],
            "pretrain_wall_seconds": seconds, "run_directory": str(run_dir),
            "dataset_info": info,
            "nonmethod_args": {key: value for key, value in args.items()
                               if key not in ("seeds", "input_mask_mode")}}


def execute(command, repo, log_path, lock_fd=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               DGLBACKEND="pytorch", WANDB_MODE="disabled", MPLBACKEND="Agg")
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT,
                                pass_fds=() if lock_fd is None else (lock_fd,))
    if result.returncode:
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-55:]), flush=True)
        raise RuntimeError("Child exited %d; log=%s" % (result.returncode, log_path))


def save_summary(batch, rows):
    groups = []
    for dataset in CONFIGS:
        for method in METHODS:
            values = [row["best_val_acc"] * 100 for row in rows
                      if row["dataset"] == dataset and row["method"] == method]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            groups.append({"dataset": dataset, "method": method, "n": len(values),
                           "best_val_acc_percent_mean": mean,
                           "best_val_acc_percent_population_std": std})
    write_json(batch / "validation_summary.json", {"groups": groups, "runs": rows,
                                                   "test_metrics_opened": False})
    for group in groups:
        print("VAL_SUMMARY %s %s n=%d %.3f +/- %.3f %%" % (
            group["dataset"], group["method"], group["n"],
            group["best_val_acc_percent_mean"], group["best_val_acc_percent_population_std"]), flush=True)


def run(cli):
    require(Path(sys.prefix).resolve() == Path("/home/jovyan/work/envs/graphmae2").resolve(),
            "Activate /home/jovyan/work/envs/graphmae2")
    repo, batch = cli.repo.resolve(), cli.batch_dir.resolve()
    require(batch.parent == repo / "outputs" and batch.name.startswith("v2_reference18_"),
            "Use a new outputs/v2_reference18_* batch directory")
    require(batch.is_dir() and {p.name for p in batch.iterdir()} <= {"batch.log", "batch.pid"},
            "Batch directory must be new; existing runs are never overwritten")
    import fcntl
    with (repo / "outputs/.formal_record_v1.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another experiment holds the batch lock")
        manifest = check_sources(repo)
        check_stage1(repo, cli.stage1_dir)
        require(not cli.checkpoint_root.resolve().is_relative_to(repo),
                "Checkpoint root must be outside the Git project")
        print("REFERENCE18_DIR=" + str(batch), flush=True)
        print("PLAN=3 datasets x B0/B2 x seeds 0,1,2; full official epoch budgets", flush=True)
        shutil.copyfile(Path(__file__).resolve(), batch / "runner_source.py")
        write_json(batch / "plan.json", {"runs": combinations(), "configs": CONFIGS,
                   "common": COMMON, "source_sha256": manifest["after_sha256"],
                   "runner_sha256": sha256(__file__), "manifest_sha256": MANIFEST_SHA256,
                   "stage1_dir": str(cli.stage1_dir.resolve()),
                   "test_metrics_used_for_selection": False})
        state = {"status": "running", "completed": 0, "total": 18,
                 "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(batch / "batch_status.json", state)
        started, rows, anchors = time.perf_counter(), [], {}
        try:
            for dataset, method, seed in combinations():
                check_sources(repo)
                name = "%s_%s_%s_s%d" % (batch.name, dataset, method, seed)
                run_dir = batch / name
                print("START dataset=%s method=%s seed=%d" % (dataset, method, seed), flush=True)
                print("TRAIN_LOG=" + str(run_dir / "train.log"), flush=True)
                execute([sys.executable, "-u", "record_full_batch_v2.py", "run", "--dataset", dataset,
                         "--method", method, "--seed", str(seed), "--phase", "development",
                         "--device", "0", "--output-dir", str(run_dir),
                         "--checkpoint-root", str(cli.checkpoint_root.resolve())],
                        repo, batch / (name + "_launch.log"), lock.fileno())
                row = validate_run(run_dir, dataset, method, seed)
                if dataset in anchors:
                    for key in ("dataset_info", "nonmethod_args"):
                        require(row[key] == anchors[dataset][key],
                                "Paired-run input/config mismatch: " + dataset + " " + key)
                else:
                    anchors[dataset] = row
                execute([sys.executable, "-u", "record_full_batch_v2.py", "verify",
                         "--output-dir", str(run_dir)], repo, run_dir / "reload.log", lock.fileno())
                require("MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS" in
                        (run_dir / "reload.log").read_text(encoding="utf-8"), "CPU reload marker missing")
                rows.append(row)
                state["completed"] = len(rows)
                write_json(batch / "batch_status.json", state)
                write_json(batch / "validated_runs.json", rows)
                print("DONE %d/18 dataset=%s method=%s seed=%d BestVal=%.4f PretrainSeconds=%.2f" % (
                    len(rows), dataset, method, seed, row["best_val_acc"], row["pretrain_wall_seconds"]), flush=True)
            check_sources(repo)
            save_summary(batch, rows)
            state.update(status="complete", wall_seconds=time.perf_counter() - started)
            write_json(batch / "batch_status.json", state)
            print("ALL_DONE 18/18", flush=True)
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__, error=str(error))
            write_json(batch / "batch_status.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("/home/jovyan/work/GraphMAE2_artifacts/checkpoints"))
    cli = parser.parse_args()
    code = 0
    try:
        run(cli)
    except BaseException as error:
        print("REFERENCE18_ERROR=" + str(error), flush=True)
        code = 1
    print("BATCH_EXIT_CODE=" + str(code), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
