"""Recompute declared overlay tree hashes after an intentional component change."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


try:
    from audit_tree import tree_sha256
    from build_release_artifacts import ReleaseArtifactError, validate_overlay_tree
except ImportError:  # pragma: no cover - only used when imported through importlib specs
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_tree import tree_sha256
    from build_release_artifacts import ReleaseArtifactError, validate_overlay_tree


MANIFEST_NAME = "overlay-manifest.yaml"
SECTIONS = ("components", "assets", "tools")


class ManifestUpdateError(RuntimeError):
    """Raised when an overlay manifest cannot be recomputed safely."""


def recompute_manifest(overlay: Path, *, version: str | None = None) -> tuple[dict[str, Any], bool]:
    manifest_path = overlay / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ManifestUpdateError(f"overlay manifest does not exist: {manifest_path}")
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestUpdateError(f"cannot read overlay manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ManifestUpdateError("overlay manifest must be schema_version 2")

    changed = version is not None and manifest.get("overlay_version") != version
    if version is not None:
        manifest["overlay_version"] = version
    for section in SECTIONS:
        nodes = manifest.get(section) or {}
        if not isinstance(nodes, dict):
            raise ManifestUpdateError(f"overlay {section} must be a mapping")
        for relative, node in nodes.items():
            if not isinstance(node, dict):
                raise ManifestUpdateError(f"overlay node {relative} must be a mapping")
            node_root = (overlay / str(relative)).resolve(strict=False)
            if overlay.resolve(strict=False) not in node_root.parents or not node_root.is_dir():
                raise ManifestUpdateError(f"overlay node is missing or escapes root: {relative}")
            digest = tree_sha256(node_root)
            if node.get("tree_sha256") != digest:
                node["tree_sha256"] = digest
                changed = True

    try:
        validate_overlay_tree(overlay.resolve(), manifest)
    except ReleaseArtifactError as exc:
        raise ManifestUpdateError(str(exc)) from exc
    return manifest, changed


def write_manifest(overlay: Path, manifest: dict[str, Any]) -> Path:
    path = overlay / MANIFEST_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=Path("overlay"))
    parser.add_argument("--version", default=None, help="set overlay_version before recomputing")
    parser.add_argument("--check", action="store_true", help="fail instead of writing when hashes differ")
    args = parser.parse_args(argv)

    try:
        manifest, changed = recompute_manifest(args.overlay, version=args.version)
    except ManifestUpdateError as exc:
        parser.error(str(exc))
    if args.check:
        if changed:
            print("overlay manifest is stale; run without --check to update it", file=sys.stderr)
            return 1
        return 0
    write_manifest(args.overlay, manifest)
    print(f"overlay manifest {'updated' if changed else 'unchanged'}: {args.overlay / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
