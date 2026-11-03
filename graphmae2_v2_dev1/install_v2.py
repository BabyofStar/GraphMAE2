"""Verify the reviewed preimage, snapshot it, and apply the v2 patch once."""

import argparse
import contextlib
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise RuntimeError("Unsafe relative filename: " + relative)
    result = root.joinpath(*name.parts)
    if result.is_symlink() or not result.resolve().is_relative_to(root):
        raise RuntimeError("Linked/outside project file: " + relative)
    return result


def verify_files(root, hashes):
    failures = []
    for name, expected in hashes.items():
        path = target(root, name)
        if not path.is_file() or sha256(path) != expected:
            failures.append(name)
    return failures


@contextlib.contextmanager
def project_lock(repo):
    outputs = Path(repo).resolve() / "outputs"
    if not outputs.resolve().is_relative_to(Path(repo).resolve()):
        raise RuntimeError("outputs resolves outside the project")
    outputs.mkdir(exist_ok=True)
    # Windows is used only for local patch/installer tests. Training targets Linux.
    if sys.platform == "win32":
        yield
        return
    import fcntl
    with (outputs / ".formal_record_v1.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A recorded experiment is running; source installation stopped")
        yield


def install(repo, package):
    with project_lock(repo):
        return install_locked(repo, package)


def install_locked(repo, package):
    repo, package = Path(repo).resolve(), Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    patch = target(package, manifest["patch_file"])
    if not patch.is_file() or sha256(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Package patch checksum mismatch")
    actual_root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True).strip()).resolve()
    if actual_root != repo:
        raise RuntimeError("--repo must be the actual Git project root")
    if not verify_files(repo, manifest["after_sha256"]):
        print("V2_ALREADY_INSTALLED=PASS")
        return None
    failures = verify_files(repo, manifest["before_sha256"])
    if failures:
        raise RuntimeError("Preimage differs from reviewed GitHub source; stop: " + ", ".join(failures))
    for name in manifest["new_files"]:
        if target(repo, name).exists():
            raise RuntimeError("A new v2 file already exists; preserve it and stop: " + name)
    command = ["git", "-c", "core.autocrlf=false", "apply", "--whitespace=error-all"]
    subprocess.run(command + ["--check", str(patch)], cwd=repo, check=True)
    outputs = repo / "outputs"
    outputs.mkdir(exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="v2_before_install_", dir=outputs))
    with tarfile.open(snapshot / "source_before_v2.tar", "x") as archive:
        for name in manifest["before_sha256"]:
            archive.add(target(repo, name), arcname=name, recursive=False)
    info = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reviewed_github_commit": manifest["reviewed_github_commit"],
        "server_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "server_git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "source_before_sha256": manifest["before_sha256"],
    }
    (snapshot / "snapshot.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_BACKUP=" + str(snapshot), flush=True)
    subprocess.run(command + [str(patch)], cwd=repo, check=True)
    failures = verify_files(repo, manifest["after_sha256"])
    if failures:
        raise RuntimeError("Post-install checksum mismatch; stop: " + ", ".join(failures))
    print("V2_SOURCE_INSTALL=PASS")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    try:
        install(args.repo, Path(__file__).resolve().parent)
    except Exception as error:
        print("V2_INSTALL_ERROR=" + str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
