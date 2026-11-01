"""Fail-closed verification for stripped official source snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

SNAPSHOT_MANIFEST_SCHEMA = "rotation_matrix_v2_snapshot_manifest_v1"
SNAPSHOT_LOCK_SCHEMA = "rotation_matrix_v2_snapshot_lock_v1"
SNAPSHOT_NAMES = ("pointmae", "pcpmae")


class SnapshotVerificationError(RuntimeError):
    """Raised when source provenance cannot be proven exactly."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _framed_update(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(len(value).to_bytes(16, "big"))
    digest.update(value)


def content_tree_inventory(root: Path) -> tuple[list[dict[str, object]], str]:
    """Hash one stripped tree without following links or accepting special files."""
    tree_root = Path(root)
    if not tree_root.is_dir():
        raise SnapshotVerificationError("snapshot root is not a directory")
    git_entry = tree_root / ".git"
    if git_entry.exists() or git_entry.is_symlink():
        raise SnapshotVerificationError("snapshot still contains .git")
    entries: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for directory, directory_names, filenames in os.walk(
        tree_root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            child = directory_path / name
            if child.is_symlink():
                raise SnapshotVerificationError(
                    "snapshot contains a directory symlink: %s"
                    % child.relative_to(tree_root)
                )
        directory_names.sort()
        for name in sorted(filenames):
            path = directory_path / name
            relative = path.relative_to(tree_root).as_posix()
            if path.is_symlink():
                raise SnapshotVerificationError(
                    "snapshot contains a file symlink: %s" % relative
                )
            if not path.is_file():
                raise SnapshotVerificationError(
                    "snapshot contains a special file: %s" % relative
                )
            stat = path.stat()
            entry = {
                "path": relative,
                "bytes": int(stat.st_size),
                "executable": bool(stat.st_mode & 0o111),
                "sha256": file_sha256(path),
            }
            entries.append(entry)
            _framed_update(digest, canonical_json_bytes(entry))
    if not entries:
        raise SnapshotVerificationError("snapshot tree is empty")
    return entries, digest.hexdigest()


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise SnapshotVerificationError("required JSON file is absent: %s" % path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotVerificationError(
            "could not parse required JSON file: %s" % path
        ) from error
    if not isinstance(value, Mapping):
        raise SnapshotVerificationError("JSON root is not a mapping: %s" % path)
    return value


def verify_snapshot(
    name: str,
    *,
    snapshots_root: Path | None = None,
) -> dict[str, object]:
    """Verify lock, commit, Git tree hash record, and every stripped file."""
    if name not in SNAPSHOT_NAMES:
        raise ValueError("unknown snapshot name: %s" % name)
    root = (
        Path(snapshots_root)
        if snapshots_root is not None
        else Path(__file__).resolve().parent / "snapshots"
    )
    lock_path = root / "SNAPSHOT_LOCK.json"
    manifest_path = root / "manifest.json"
    lock = _load_json_mapping(lock_path)
    manifest = _load_json_mapping(manifest_path)
    if lock.get("schema") != SNAPSHOT_LOCK_SCHEMA:
        raise SnapshotVerificationError("snapshot lock schema mismatch")
    if manifest.get("schema") != SNAPSHOT_MANIFEST_SCHEMA:
        raise SnapshotVerificationError("snapshot manifest schema mismatch")
    if manifest.get("materialized") is not True:
        raise SnapshotVerificationError("snapshot manifest is not materialized")
    lock_repositories = lock.get("repositories")
    repositories = manifest.get("repositories")
    if not isinstance(lock_repositories, Mapping) or not isinstance(
        repositories, Mapping
    ):
        raise SnapshotVerificationError("snapshot repository records are invalid")
    if set(lock_repositories) != set(SNAPSHOT_NAMES):
        raise SnapshotVerificationError("snapshot lock repository set mismatch")
    if set(repositories) != set(SNAPSHOT_NAMES):
        raise SnapshotVerificationError("snapshot manifest repository set mismatch")
    expected = lock_repositories[name]
    record = repositories[name]
    if not isinstance(expected, Mapping) or not isinstance(record, Mapping):
        raise SnapshotVerificationError("snapshot entry is not a mapping")
    for key in ("repository", "url", "commit_sha", "directory"):
        if record.get(key) != expected.get(key):
            raise SnapshotVerificationError(
                "snapshot %s differs from lock field %s" % (name, key)
            )
    git_tree = record.get("git_tree_sha1")
    if (
        not isinstance(git_tree, str)
        or len(git_tree) != 40
        or any(character not in "0123456789abcdef" for character in git_tree)
    ):
        raise SnapshotVerificationError("snapshot Git tree hash is invalid")
    expected_files = record.get("files")
    expected_content_hash = record.get("content_tree_sha256")
    if not isinstance(expected_files, list) or not isinstance(
        expected_content_hash, str
    ):
        raise SnapshotVerificationError("snapshot content inventory is absent")
    snapshot_path = root / str(record["directory"])
    actual_files, actual_content_hash = content_tree_inventory(snapshot_path)
    if actual_files != expected_files:
        raise SnapshotVerificationError("snapshot file inventory changed")
    if actual_content_hash != expected_content_hash:
        raise SnapshotVerificationError("snapshot content tree hash changed")
    result = dict(record)
    result["path"] = str(snapshot_path.resolve())
    result["manifest_path"] = str(manifest_path.resolve())
    result["manifest_sha256"] = file_sha256(manifest_path)
    result["lock_sha256"] = file_sha256(lock_path)
    result["verified"] = True
    return result
