"""Run D x 9, then S x 9, paired against the completed v2 reference batch."""

import argparse
import datetime
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True

REFERENCE_RUNNER_SHA256 = "ebfa750def8e66ae26deec251a1de830b52a41f736e682f818adedca1928d603"
METHODS = {"D": "reconstruction_curriculum", "S": "density_stratified"}
ENVIRONMENT_KEYS = ("versions", "torch_cuda_runtime", "torch_cpu_threads",
                    "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
                    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "gpu_name",
                    "visible_gpu_memory_bytes")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_library(path):
    require(sha256(path) == REFERENCE_RUNNER_SHA256,
            "Reference queue source differs from the approved version")
    spec = importlib.util.spec_from_file_location("_approved_v2_reference18", path)
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    return library


def combinations(library):
    return [(dataset, method, seed) for method in METHODS
            for dataset in library.CONFIGS for seed in (0, 1, 2)]


def environment_signature(library, run_dir):
    environment = library.read_json(run_dir / "environment.json")
    require(all(key in environment for key in ENVIRONMENT_KEYS), "Incomplete environment record")
    return {key: environment[key] for key in ENVIRONMENT_KEYS}


def load_references(library, repo, directory, manifest):
    require(directory.parent == repo / "outputs", "Reference batch must be in project outputs")
    state = library.read_json(directory / "batch_status.json")
    require((state.get("status"), state.get("completed"), state.get("total")) ==
            ("complete", 18, 18), "The 18 reference runs are not complete")
    plan = library.read_json(directory / "plan.json")
    require(plan["source_sha256"] == manifest["after_sha256"], "Reference training sources differ")
    require(plan["runner_sha256"] == REFERENCE_RUNNER_SHA256, "Reference runner identity differs")
    require(plan["runs"] == [list(job) for job in library.combinations()], "Reference combinations differ")
    anchors, rows = {}, []
    for dataset, method, seed in library.combinations():
        run_dir = directory / ("%s_%s_%s_s%d" % (directory.name, dataset, method, seed))
        row = library.validate_run(run_dir, dataset, method, seed)
        require("MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS" in
                (run_dir / "reload.log").read_text(encoding="utf-8"), "Reference CPU reload marker missing")
        row["environment_signature"] = environment_signature(library, run_dir)
        key = dataset, seed
        if key not in anchors:
            anchors[key] = {}
        anchors[key][method] = row
        rows.append(row)
    for paired in anchors.values():
        for name in ("dataset_info", "nonmethod_args", "environment_signature"):
            require(paired["B0"][name] == paired["B2"][name], "Reference pair mismatch: " + name)
    return anchors, rows


def validate_layers(item, n, k, stratified):
    layers = item.get("density_layers")
    require(isinstance(layers, list) and len(layers) == 4, "Four density layers are required")
    sizes = [n // 4 + (i < n % 4) for i in range(4)]
    require([row["layer"] for row in layers] == list(range(4)) and
            [row["nodes"] for row in layers] == sizes, "Density layer sizes/indices differ")
    masked = [row["masked"] for row in layers]
    require(sum(masked) == k and all(0 <= count <= size for count, size in zip(masked, sizes)),
            "Layer mask counts differ from the total budget")
    if stratified:
        # Independent exact-rational check of the advertised proportional quotas.
        ideals = [Fraction(k * size, n) for size in sizes]
        quotas = [math.floor(value) for value in ideals]
        order = sorted(range(4), key=lambda i: (-(ideals[i] - quotas[i]), i))
        for i in order[:k - sum(quotas)]:
            quotas[i] += 1
        require(masked == quotas, "Density-stratified quotas differ")


def validate_candidate(library, run_dir, dataset, method, seed, reference):
    status, metrics, args, resources = [library.read_json(run_dir / name) for name in
        ("status.json", "metrics.json", "effective_args.json", "resources.json")]
    for record in (status, metrics):
        for key, expected in {"status": "complete", "phase": "development", "dataset": dataset,
                              "method": method, "seed": seed, "official_epoch_budget": True}.items():
            require(record.get(key) == expected, "Candidate metadata differs: " + key)
    require(args["seeds"] == [seed] and args["input_mask_mode"] == METHODS[method],
            "Candidate seed or input-mask mode differs")
    nonmethod = {key: value for key, value in args.items() if key not in ("seeds", "input_mask_mode")}
    require(nonmethod == reference["nonmethod_args"], "Candidate nonmethod configuration differs from B0/B2")
    info = metrics["dataset_info"]
    require(info == reference["dataset_info"], "Candidate graph/features/splits differ from B0/B2")
    environment = environment_signature(library, run_dir)
    require(environment == reference["environment_signature"], "Candidate runtime settings differ from B0/B2")
    val, epoch = metrics["best_val_acc"], metrics["best_val_epoch"]
    require(isinstance(val, (int, float)) and math.isfinite(val) and 0 <= val <= 1,
            "Invalid candidate validation accuracy")
    require(isinstance(epoch, int) and 0 <= epoch < args["max_epoch_f"], "Invalid classifier epoch")
    epochs = args["max_epoch"]
    expected_steps = sorted(set(range(0, epochs, 100)) | {epochs - 1})
    feedback = [json.loads(line) for line in (run_dir / "mask_feedback.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    require([row["step"] for row in feedback] == expected_steps and
            metrics["feedback_records"] == len(expected_steps), "Candidate feedback steps differ")
    n, k = info["nodes"], int(info["nodes"] * args["mask_rate"])
    for item in feedback:
        require(item["mode"] == METHODS[method] and item["nodes"] == n and item["masked"] == k,
                "Candidate mask mode or node budget differs")
        progress = item["step"] / max(epochs - 1, 1)
        alpha = 0.0 if method == "S" else 0.5 * min(max((progress - 0.2) / (0.8 - 0.2), 0), 1)
        require(math.isclose(item["alpha"], alpha, abs_tol=1e-12), "Candidate curriculum schedule differs")
        require(math.isclose(item["observation_count_min_mean_max"][1],
                             k * (item["step"] + 1) / n, rel_tol=1e-9, abs_tol=1e-8),
                "Candidate observation budget differs")
        validate_layers(item, n, k, method == "S")
    node_path = run_dir / metrics["mask_node_statistics_file"]
    require(library.sha256(node_path) == metrics["mask_node_statistics_sha256"],
            "Candidate node-statistic checksum differs")
    seconds = resources["pretrain_including_intermediate_probes"]["wall_seconds"]
    require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0,
            "Invalid candidate resource record")
    # No test_metrics.json is opened here or in the summary code.
    return {"dataset": dataset, "method": method, "seed": seed, "best_val_acc": val,
            "best_val_epoch": epoch, "pretrain_wall_seconds": seconds,
            "run_directory": str(run_dir), "dataset_info": info,
            "nonmethod_args": nonmethod, "environment_signature": environment,
            "last_mask_feedback": feedback[-1]}


def save_summary(library, directory, rows, anchors):
    groups = []
    for dataset in library.CONFIGS:
        for method in METHODS:
            paired = sorted([row for row in rows if row["dataset"] == dataset
                             and row["method"] == method], key=lambda row: row["seed"])
            require([row["seed"] for row in paired] == [0, 1, 2], "Incomplete candidate group")
            values = [100 * row["best_val_acc"] for row in paired]
            mean = sum(values) / 3
            result = {"dataset": dataset, "method": method, "n": 3,
                      "best_val_acc_percent_mean": mean,
                      "best_val_acc_percent_population_std": math.sqrt(
                          sum((value - mean) ** 2 for value in values) / 3)}
            for reference in ("B0", "B2"):
                differences = [100 * (row["best_val_acc"] -
                    anchors[dataset, row["seed"]][reference]["best_val_acc"]) for row in paired]
                result["paired_val_pp_vs_" + reference] = differences
                result["mean_val_pp_vs_" + reference] = sum(differences) / 3
            groups.append(result)
            print("VAL_COMPARE %s %s n=3 %.3f +/- %.3f %% delta_B0=%+.3fpp delta_B2=%+.3fpp" % (
                dataset, method, mean, result["best_val_acc_percent_population_std"],
                result["mean_val_pp_vs_B0"], result["mean_val_pp_vs_B2"]), flush=True)
    library.write_json(directory / "validation_summary.json", {"groups": groups, "runs": rows,
        "test_metrics_opened": False, "interpretation": "development validation only, n=3"})


def run(cli):
    require(Path(sys.prefix).resolve() == Path("/home/jovyan/work/envs/graphmae2").resolve(),
            "Activate /home/jovyan/work/envs/graphmae2")
    repo, directory, references = cli.repo.resolve(), cli.batch_dir.resolve(), cli.reference_dir.resolve()
    require(directory.parent == repo / "outputs" and directory.name.startswith("v2_candidates18_"),
            "Use a new project outputs/v2_candidates18_* directory")
    require(directory.is_dir() and {path.name for path in directory.iterdir()} <= {"batch.log", "batch.pid"},
            "Batch must be new; no previous runs will be overwritten")
    require(references.parent == repo / "outputs", "Reference batch must be in project outputs")
    library = load_reference_library(references / "runner_source.py")
    import fcntl
    with (repo / "outputs/.formal_record_v1.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another experiment holds the batch lock")
        manifest = library.check_sources(repo)
        anchors, reference_rows = load_references(library, repo, references, manifest)
        require(not cli.checkpoint_root.resolve().is_relative_to(repo), "Keep weights outside the Git project")
        print("CANDIDATES18_DIR=" + str(directory), flush=True)
        print("REFERENCE_AUDIT=PASS 18/18; validation only", flush=True)
        print("PLAN=D x 9, then S x 9; official full budgets; no DS", flush=True)
        shutil.copyfile(__file__, directory / "runner_source.py")
        shutil.copyfile(references / "runner_source.py", directory / "reference_runner_source.py")
        library.write_json(directory / "reference_validation_rows.json", reference_rows)
        library.write_json(directory / "plan.json", {"runs": combinations(library),
            "reference_dir": str(references), "source_sha256": manifest["after_sha256"],
            "runner_sha256": sha256(__file__), "reference_runner_sha256": REFERENCE_RUNNER_SHA256,
            "settings": "frozen v2-dev1; official epoch budgets; seeds 0,1,2",
            "test_metrics_used_for_selection": False})
        state = {"status": "running", "completed": 0, "total": 18,
                 "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        library.write_json(directory / "batch_status.json", state)
        started, rows = time.perf_counter(), []
        try:
            for dataset, method, seed in combinations(library):
                library.check_sources(repo)
                name = "%s_%s_%s_s%d" % (directory.name, dataset, method, seed)
                run_dir = directory / name
                print("START dataset=%s method=%s seed=%d" % (dataset, method, seed), flush=True)
                print("TRAIN_LOG=" + str(run_dir / "train.log"), flush=True)
                library.execute([sys.executable, "-u", "record_full_batch_v2.py", "run",
                    "--dataset", dataset, "--method", method, "--seed", str(seed),
                    "--phase", "development", "--device", "0", "--output-dir", str(run_dir),
                    "--checkpoint-root", str(cli.checkpoint_root.resolve())],
                    repo, directory / (name + "_launch.log"), lock.fileno())
                row = validate_candidate(library, run_dir, dataset, method, seed, anchors[dataset, seed]["B0"])
                library.execute([sys.executable, "-u", "record_full_batch_v2.py", "verify",
                    "--output-dir", str(run_dir)], repo, run_dir / "reload.log", lock.fileno())
                require("MODEL_AND_CLASSIFIERS_CPU_RELOAD=PASS" in
                        (run_dir / "reload.log").read_text(encoding="utf-8"), "CPU reload marker missing")
                rows.append(row)
                state["completed"] = len(rows)
                library.write_json(directory / "batch_status.json", state)
                library.write_json(directory / "validated_runs.json", rows)
                print("DONE %d/18 dataset=%s method=%s seed=%d BestVal=%.4f PretrainSeconds=%.2f" % (
                    len(rows), dataset, method, seed, row["best_val_acc"], row["pretrain_wall_seconds"]), flush=True)
            library.check_sources(repo)
            save_summary(library, directory, rows, anchors)
            state.update(status="complete", wall_seconds=time.perf_counter() - started)
            library.write_json(directory / "batch_status.json", state)
            print("ALL_DONE 18/18", flush=True)
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__, error=str(error))
            library.write_json(directory / "batch_status.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("/home/jovyan/work/GraphMAE2_artifacts/checkpoints"))
    cli = parser.parse_args()
    code = 0
    try:
        run(cli)
    except BaseException as error:
        print("CANDIDATES18_ERROR=" + str(error), flush=True)
        code = 1
    print("BATCH_EXIT_CODE=" + str(code), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
