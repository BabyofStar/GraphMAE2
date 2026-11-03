"""Export completed development metadata and masking diagnostics, without weights/test scores."""

import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile


RUNNERS = {
    "reference": "ebfa750def8e66ae26deec251a1de830b52a41f736e682f818adedca1928d603",
    "candidate": "34fa7ca3a6415c025f83a0bd37281c251660b1dd00c46a7410af154436add12f",
}
METHODS = {"reference": ("B0", "B2"), "candidate": ("D", "S")}
ROOT_FILES = ("batch.log", "batch_status.json", "plan.json", "validation_summary.json",
              "validated_runs.json", "runner_source.py")
RUN_FILES = ("status.json", "metrics.json", "effective_args.json", "environment.json",
             "resources.json", "mask_stats.jsonl", "mask_feedback.jsonl",
             "mask_node_statistics.npz", "source_sha256.json", "source.tar", "reload.log")
EPOCHS = {"cora": (2000, 300), "citeseer": (500, 500), "pubmed": (2000, 500)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete(directory):
    path = directory / "batch_status.json"
    if not path.is_file():
        return False
    state = read_json(path)
    return (state.get("status"), state.get("completed"), state.get("total")) == ("complete", 18, 18)


def choose_candidate(repo, explicit=None):
    if explicit is not None:
        result = Path(explicit).resolve()
        require(result.parent == repo / "outputs" and complete(result),
                "--candidate-dir must be a completed project outputs batch")
        return result
    matches = []
    for directory in sorted((repo / "outputs").glob("v2_candidates18_*")):
        if complete(directory):
            matches.append(directory.resolve())
    require(len(matches) == 1, "Expected one completed candidate batch; rerun with --candidate-dir. Found: "
            + ", ".join(str(path) for path in matches))
    return matches[0]


def collect(repo, candidate):
    require(candidate.parent == repo / "outputs", "Candidate must be in project outputs")
    candidate_plan = read_json(candidate / "plan.json")
    reference = Path(candidate_plan["reference_dir"]).resolve()
    require(reference.parent == repo / "outputs", "Reference must be in project outputs")
    roots = {"reference": reference, "candidate": candidate}
    members, combinations, frozen_source = [], [], None
    for role, directory in roots.items():
        require(complete(directory), "Batch is not complete: " + str(directory))
        plan = read_json(directory / "plan.json")
        require(plan["runner_sha256"] == RUNNERS[role]
                and sha256(directory / "runner_source.py") == RUNNERS[role],
                "Unexpected batch runner: " + role)
        if frozen_source is None:
            frozen_source = plan["source_sha256"]
        require(plan["source_sha256"] == frozen_source, "Reference/candidate source versions differ")
        expected = {(dataset, method, seed) for dataset in EPOCHS
                    for method in METHODS[role] for seed in (0, 1, 2)}
        jobs = [tuple(job) for job in plan["runs"]]
        require(len(jobs) == 18 and set(jobs) == expected, "Unexpected run combinations: " + role)
        for name in ROOT_FILES:
            members.append((directory / name, role + "/" + name, directory))
        for name in ("reference_runner_source.py", "reference_validation_rows.json"):
            if (directory / name).is_file():
                members.append((directory / name, role + "/" + name, directory))
        for dataset, method, seed in jobs:
            run_dir = directory / ("%s_%s_%s_s%d" % (directory.name, dataset, method, seed))
            status, metrics, args = [read_json(run_dir / name) for name in
                                     ("status.json", "metrics.json", "effective_args.json")]
            for record in (status, metrics):
                require((record.get("status"), record.get("phase"), record.get("dataset"),
                         record.get("method"), record.get("seed"), record.get("official_epoch_budget"))
                        == ("complete", "development", dataset, method, seed, True),
                        "Incomplete or mismatched run: " + str(run_dir))
            require((args["max_epoch"], args["max_epoch_f"]) == EPOCHS[dataset],
                    "Not the official complete epoch budget: " + str(run_dir))
            short_name = "%s_%s_s%d" % (dataset, method, seed)
            for name in RUN_FILES:
                members.append((run_dir / name, role + "/runs/" + short_name + "/" + name, directory))
            combinations.append({"role": role, "dataset": dataset, "method": method,
                                 "seed": seed, "server_directory": str(run_dir)})
    for path, name, directory in members:
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(directory), "Missing or linked result file: " + str(path))
        require(path.name != "test_metrics.json" and path.suffix not in (".pt", ".pth"),
                "Excluded result type: " + name)
    return members, {"format_version": 1, "runs": combinations,
                     "batch_directories": {key: str(value) for key, value in roots.items()},
                     "test_scores_included": False, "model_weights_included": False}


def export(repo, explicit=None):
    repo = Path(repo).resolve()
    candidate = choose_candidate(repo, explicit)
    members, index = collect(repo, candidate)
    directory = Path(tempfile.mkdtemp(prefix="v2_development36_export_", dir=repo / "outputs"))
    archive = directory / "v2_development36_results.tar.gz"
    index["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    index["files"] = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                      for path, name, _ in members}
    with tarfile.open(archive, "x:gz") as stream:
        for path, name, _ in members:
            stream.add(path, arcname=name, recursive=False)
            require(sha256(path) == index["files"][name]["sha256"],
                    "A result file changed during export: " + str(path))
        data = (json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        item = tarfile.TarInfo("export_manifest.json")
        item.size = len(data)
        stream.addfile(item, io.BytesIO(data))
    digest = sha256(archive)
    (directory / "SHA256SUMS.txt").write_text(digest + "  " + archive.name + "\n", encoding="utf-8")
    print("CANDIDATE_DIR=" + str(candidate))
    print("EXPORTED_RUNS=" + str(len(index["runs"])))
    print("ARCHIVE=" + str(archive))
    print("ARCHIVE_BYTES=" + str(archive.stat().st_size))
    print("ARCHIVE_SHA256=" + digest)
    print("DEVELOPMENT36_EXPORT=PASS")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/jovyan/work/GraphMAE2"))
    parser.add_argument("--candidate-dir", type=Path)
    args = parser.parse_args()
    try:
        export(args.repo, args.candidate_dir)
    except Exception as error:
        print("DEVELOPMENT36_EXPORT_ERROR=" + str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
