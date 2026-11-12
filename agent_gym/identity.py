"""Reproducibility identities for framework, overlay and AgentGym repository."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from agent_gym.layout import GymLayout
from agent_gym.overlay import OverlayError, load_manifest, sha256_file, tree_sha256


ENVIRONMENT_IDENTITY_SCHEMA_VERSION = 2
ENVIRONMENT_DIGEST_PREFIX = "sha256:"


def framework_identity(layout: GymLayout | None = None) -> dict[str, Any]:
    try:
        version = importlib.metadata.version("nemo-gym")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    spec = importlib.util.find_spec("nemo_gym")
    location: str | None = None
    if spec is not None:
        if spec.submodule_search_locations:
            location = str(Path(next(iter(spec.submodule_search_locations))))
        elif spec.origin:
            location = str(Path(spec.origin).parent)
    identity: dict[str, Any] = {
        "distribution": "nemo-gym",
        "version": version,
        "file": location,
    }
    if layout is not None:
        identity["expected_root"] = str(layout.framework_root)
    return identity


def repo_identity(repo_root: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        commit = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {"root": str(repo_root), "commit": commit}


def overlay_identity(layout: GymLayout) -> dict[str, Any]:
    if not layout.manifest_path.is_file():
        raise OverlayError(f"overlay manifest does not exist: {layout.manifest_path}")
    manifest = load_manifest(layout.manifest_path)
    return {
        "root": str(layout.overlay_root),
        "manifest": str(layout.manifest_path),
        "manifest_sha256": sha256_file(layout.manifest_path),
        "version": manifest.get("overlay_version"),
        "framework_version": (manifest.get("framework") or {}).get("version"),
    }


def configuration_identity(layout: GymLayout) -> dict[str, Any]:
    return {
        "framework": framework_identity(layout),
        "overlay": overlay_identity(layout),
        "repo": repo_identity(layout.repo_root),
        "workspace": str(layout.workspace_root),
    }


def _source_tree_sha256(path: Path) -> str | None:
    try:
        return tree_sha256(path) if path.is_dir() else None
    except (OSError, OverlayError):
        return None


def environment_snapshot(layout: GymLayout, *, repo_commit: str | None = None) -> dict[str, Any]:
    """Return the portable, path-free identity of the runtime that produces scores.

    Unlike :func:`configuration_identity`, this snapshot deliberately excludes absolute paths.
    It is safe to hash and compare across hosts.  The installed framework version and the
    declared wheel/package hashes are both recorded: the former catches upgrades, while the
    latter catches rebuilt wheels or a manifest that no longer matches the installed framework.
    """
    manifest = load_manifest(layout.manifest_path) if layout.manifest_path.is_file() else None
    framework_manifest = (manifest or {}).get("framework") or {}
    installed_framework = framework_identity(layout)
    repo = repo_identity(layout.repo_root)
    evaluator_package = Path(__file__).resolve().parent
    eval_service_package = evaluator_package.parent / "eval_service"

    return {
        "schema_version": ENVIRONMENT_IDENTITY_SCHEMA_VERSION,
        "repo": {
            "commit": repo_commit or repo.get("commit") or "unknown",
        },
        "framework": {
            "distribution": installed_framework.get("distribution", "nemo-gym"),
            "version": installed_framework.get("version", "unknown"),
            "declared_version": framework_manifest.get("version"),
            "wheel_sha256": framework_manifest.get("wheel_sha256"),
            "package_tree_sha256": framework_manifest.get("package_tree_sha256"),
        },
        "overlay": {
            "version": (manifest or {}).get("overlay_version"),
            "manifest_sha256": sha256_file(layout.manifest_path) if layout.manifest_path.is_file() else None,
        },
        "evaluator": {
            "agent_gym_source_sha256": _source_tree_sha256(evaluator_package),
            "eval_service_source_sha256": _source_tree_sha256(eval_service_package),
        },
        "verified": bool(manifest and manifest.get("schema_version") == 2),
    }


def environment_digest(snapshot: dict[str, Any]) -> str:
    """Hash a runtime snapshot without machine-local paths or verification state."""
    payload = {
        "schema_version": snapshot.get("schema_version"),
        "repo": snapshot.get("repo"),
        "framework": snapshot.get("framework"),
        "overlay": snapshot.get("overlay"),
        "evaluator": snapshot.get("evaluator"),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return ENVIRONMENT_DIGEST_PREFIX + hashlib.sha256(canonical).hexdigest()
