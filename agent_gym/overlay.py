"""Overlay manifest loading and deterministic tree integrity checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import yaml

from agent_gym.layout import OVERLAY_MANIFEST, GymLayout


class OverlayError(RuntimeError):
    """Raised when an overlay manifest or tree is invalid."""


ENV_NEMO_WHEEL = "AGENT_GYM_NEMO_WHEEL"
IGNORED_TREE_DIRS = {
    ".git",
    ".venv",
    ".baseline",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".tau2_data.lockdir",
    "__pycache__",
    "cache",
    "data",
    "node_modules",
    "outputs",
    "results",
    "tau2_data",
}


def _normalize_sha256(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.removeprefix("sha256:").removeprefix("sha256=").lower()


def _installed_distribution() -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution("nemo-gym")
    except importlib.metadata.PackageNotFoundError:
        return None


def _lock_wheel_hashes(layout: GymLayout, version: str) -> set[str]:
    lock_path = layout.repo_root / "uv.lock"
    if not lock_path.is_file():
        return set()
    try:
        with open(lock_path, "rb") as handle:
            locked = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise OverlayError(f"cannot read framework lock file {lock_path}: {exc}") from exc
    hashes: set[str] = set()
    for package in locked.get("package", []):
        if package.get("name") != "nemo-gym" or package.get("version") != version:
            continue
        for wheel in package.get("wheels", []):
            digest = _normalize_sha256(wheel.get("hash"))
            if digest:
                hashes.add(digest)
    return hashes


def _wheel_hash_evidence(layout: GymLayout, version: str) -> dict[str, str]:
    evidence: dict[str, str] = {}
    explicit = os.environ.get(ENV_NEMO_WHEEL)
    if explicit:
        wheel_path = Path(explicit).expanduser()
        if not wheel_path.is_file():
            raise OverlayError(f"{ENV_NEMO_WHEEL} does not point to a wheel: {wheel_path}")
        evidence[f"{ENV_NEMO_WHEEL}:{wheel_path.name}"] = sha256_file(wheel_path)

    distribution = _installed_distribution()
    if distribution is not None:
        try:
            direct_url_text = distribution.read_text("direct_url.json")
        except (OSError, UnicodeDecodeError):
            direct_url_text = None
        if direct_url_text:
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError as exc:
                raise OverlayError(f"installed nemo-gym has invalid direct_url.json: {exc}") from exc
            archive = direct_url.get("archive_info") or {}
            digest = _normalize_sha256(archive.get("hash"))
            if digest is None:
                digest = _normalize_sha256((archive.get("hashes") or {}).get("sha256"))
            if digest:
                evidence["installed-direct-url"] = digest

    for digest in sorted(_lock_wheel_hashes(layout, version)):
        # A lock may publish more than one wheel for the same version (for
        # example, platform-specific artifacts). Keep every hash as evidence;
        # collapsing them under one key would silently discard all but the last.
        evidence[f"uv.lock:{digest}"] = digest
    return evidence


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(data: bytes) -> bytes:
    """Normalize text line endings so checkout EOL policy cannot change tree identity."""
    if b"\0" in data:
        return data
    return data.replace(b"\r\n", b"\n")


def _tree_entries(root: Path) -> list[tuple[str, str, int, bytes | str]]:
    entries: list[tuple[str, str, int, bytes | str]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if name not in IGNORED_TREE_DIRS or (current_path / name).is_symlink()
        )
        files.sort()
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                target_path = (path.parent / target).resolve(strict=False)
                if Path(target).is_absolute() or root.resolve() not in target_path.parents:
                    raise OverlayError(f"symlink escapes overlay tree: {path} -> {target}")
                entries.append((relative, "symlink", 0, Path(target).as_posix()))
            else:
                entries.append((relative, "file", 0o644, canonical_bytes(path.read_bytes())))
        for name in sorted(directories):
            path = current_path / name
            if path.is_symlink():
                target = os.readlink(path)
                target_path = (path.parent / target).resolve(strict=False)
                if Path(target).is_absolute() or root.resolve() not in target_path.parents:
                    raise OverlayError(f"symlink escapes overlay tree: {path} -> {target}")
                entries.append((path.relative_to(root).as_posix(), "symlink", 0, Path(target).as_posix()))
    entries.sort(key=lambda item: item[0])
    return entries


def tree_sha256(root: str | Path) -> str:
    path = Path(root)
    if not path.is_dir():
        raise OverlayError(f"overlay tree is not a directory: {path}")
    digest = hashlib.sha256()
    for relative, kind, mode, payload in _tree_entries(path):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(f"{mode:04o}".encode("ascii"))
        digest.update(b"\0")
        if isinstance(payload, bytes):
            digest.update(hashlib.sha256(payload).digest())
        else:
            digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise OverlayError(f"overlay manifest does not exist: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise OverlayError(f"overlay manifest must be a mapping: {manifest_path}")
    return data


def _unexpected_overlay_entries(layout: GymLayout, node_paths: set[str]) -> list[str]:
    """Return files/directories not covered by a declared manifest node."""
    unexpected: list[str] = []
    root = layout.overlay_root.resolve(strict=False)
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root).as_posix()
        if relative_dir != ".":
            covered = any(
                node == relative_dir or relative_dir.startswith(f"{node}/") or node.startswith(f"{relative_dir}/")
                for node in node_paths
            )
            if not covered:
                unexpected.append(relative_dir)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                if not any(relative == node or relative.startswith(f"{node}/") for node in node_paths):
                    unexpected.append(relative)
        for name in files:
            relative = (current_path / name).relative_to(root).as_posix()
            if relative == OVERLAY_MANIFEST:
                continue
            if not any(relative == node or relative.startswith(f"{node}/") for node in node_paths):
                unexpected.append(relative)
    return sorted(set(unexpected))


def verify_manifest(
    layout: GymLayout,
    *,
    verify_trees: bool = True,
    verify_framework: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(layout.manifest_path)
    if manifest.get("schema_version") != 2:
        raise OverlayError("overlay manifest schema_version must be 2")
    framework = manifest.get("framework") or {}
    if framework.get("distribution") != "nemo-gym":
        raise OverlayError("overlay framework.distribution must be nemo-gym")
    framework_version = framework.get("version")
    if not framework_version:
        raise OverlayError("overlay framework.version is required")
    if verify_framework:
        distribution = _installed_distribution()
        if distribution is None:
            raise OverlayError("nemo-gym distribution is not installed")
        if distribution.version != framework_version:
            raise OverlayError(
                "overlay framework.version does not match the installed nemo-gym: "
                f"manifest={framework_version}, installed={distribution.version}"
            )
        expected_wheel = _normalize_sha256(framework.get("wheel_sha256"))
        if not expected_wheel:
            raise OverlayError("overlay framework.wheel_sha256 is required")
        expected_package_tree = _normalize_sha256(framework.get("package_tree_sha256"))
        if not expected_package_tree:
            raise OverlayError("overlay framework.package_tree_sha256 is required")
        package_root = layout.framework_root / "nemo_gym"
        if not package_root.is_dir():
            raise OverlayError(f"installed nemo_gym package is missing: {package_root}")
        actual_package_tree = tree_sha256(package_root)
        if actual_package_tree != expected_package_tree:
            raise OverlayError(
                "overlay framework.package_tree_sha256 does not match the installed nemo_gym: "
                f"manifest={expected_package_tree}, installed={actual_package_tree}"
            )
        evidence = _wheel_hash_evidence(layout, framework_version)
        if not evidence:
            raise OverlayError(
                "cannot verify overlay framework.wheel_sha256: no wheel hash evidence; "
                f"set {ENV_NEMO_WHEEL} to the installed nemo-gym wheel"
            )
        if expected_wheel not in evidence.values():
            rendered = ", ".join(f"{source}={digest}" for source, digest in sorted(evidence.items()))
            raise OverlayError(
                "overlay framework.wheel_sha256 does not match the installed nemo-gym: "
                f"manifest={expected_wheel}; evidence: {rendered}"
            )
    node_paths: set[str] = set()
    for section in ("components", "assets", "tools"):
        nodes = manifest.get(section) or {}
        if not isinstance(nodes, dict):
            raise OverlayError(f"overlay {section} must be a mapping")
        for relative, node in nodes.items():
            if not isinstance(node, dict):
                raise OverlayError(f"overlay node {relative} must be a mapping")
            node_path = (layout.overlay_root / relative).resolve(strict=False)
            if layout.overlay_root.resolve() not in node_path.parents:
                raise OverlayError(f"overlay node escapes root: {relative}")
            if not node_path.is_dir():
                raise OverlayError(f"overlay node directory is missing: {relative}")
            node_paths.add(Path(relative).as_posix().rstrip("/"))
            expected = node.get("tree_sha256")
            if verify_trees:
                if not expected:
                    raise OverlayError(f"overlay node has no tree_sha256: {relative}")
                actual = tree_sha256(node_path)
                if actual != expected:
                    raise OverlayError(f"overlay tree hash mismatch for {relative}: expected {expected}, got {actual}")
    unexpected = _unexpected_overlay_entries(layout, node_paths)
    if unexpected:
        rendered = ", ".join(unexpected[:10])
        suffix = "" if len(unexpected) <= 10 else f" (+{len(unexpected) - 10} more)"
        raise OverlayError(f"overlay contains paths outside declared manifest nodes: {rendered}{suffix}")
    return manifest


def manifest_path_for(root: str | Path) -> Path:
    return Path(root) / OVERLAY_MANIFEST
