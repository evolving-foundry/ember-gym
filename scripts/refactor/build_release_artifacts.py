"""Build deterministic overlay/data artifacts for one AgentGym release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = 1
INTERNAL_TOOL_DIRS = {
    ".git",
    ".venv",
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


class ReleaseArtifactError(RuntimeError):
    """Raised when overlay/data inputs cannot produce a release artifact."""


def canonical_bytes(data: bytes) -> bytes:
    if b"\0" in data:
        return data
    return data.replace(b"\r\n", b"\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(overlay: Path) -> dict[str, Any]:
    manifest_path = overlay / "overlay-manifest.yaml"
    if not manifest_path.is_file():
        raise ReleaseArtifactError(f"overlay manifest does not exist: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise ReleaseArtifactError("overlay manifest must be schema_version 2")
    return data


def _node_paths(manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for section in ("components", "assets", "tools"):
        nodes = manifest.get(section) or {}
        if not isinstance(nodes, dict):
            raise ReleaseArtifactError(f"overlay {section} must be a mapping")
        for relative in nodes:
            normalized = Path(str(relative)).as_posix().rstrip("/")
            if not normalized or normalized == "." or Path(normalized).is_absolute() or ".." in Path(normalized).parts:
                raise ReleaseArtifactError(f"invalid overlay node path: {relative!r}")
            paths.add(normalized)
    return paths


def _tree_hash(root: Path) -> str:
    root = root.resolve(strict=False)
    entries: list[tuple[str, str, int, bytes | str]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if name not in INTERNAL_TOOL_DIRS or (current_path / name).is_symlink()
        )
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                resolved_target = (path.parent / target).resolve(strict=False)
                if Path(target).is_absolute() or root not in resolved_target.parents:
                    raise ReleaseArtifactError(f"symlink escapes overlay node: {path} -> {target}")
                entries.append((relative, "symlink", 0, Path(target).as_posix()))
            else:
                entries.append((relative, "file", 0o644, canonical_bytes(path.read_bytes())))
        for name in sorted(directories):
            path = current_path / name
            if not path.is_symlink():
                continue
            target = os.readlink(path)
            resolved_target = (path.parent / target).resolve(strict=False)
            if Path(target).is_absolute() or root not in resolved_target.parents:
                raise ReleaseArtifactError(f"symlink escapes overlay node: {path} -> {target}")
            entries.append((path.relative_to(root).as_posix(), "symlink", 0, Path(target).as_posix()))
    digest = hashlib.sha256()
    for relative, kind, mode, payload in sorted(entries, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(f"{mode:04o}".encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest() if isinstance(payload, bytes) else payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_overlay_tree(overlay: Path, manifest: dict[str, Any]) -> None:
    nodes = _node_paths(manifest)
    root = overlay.resolve(strict=False)
    for section in ("components", "assets", "tools"):
        for relative, node in (manifest.get(section) or {}).items():
            node_root = (overlay / relative).resolve(strict=False)
            if root not in node_root.parents or not node_root.is_dir():
                raise ReleaseArtifactError(f"overlay node is missing or escapes root: {relative}")
            expected = node.get("tree_sha256")
            if not expected:
                raise ReleaseArtifactError(f"overlay node has no tree_sha256: {relative}")
            actual = _tree_hash(node_root)
            if actual != expected:
                raise ReleaseArtifactError(
                    f"overlay tree hash mismatch for {relative}: expected {expected}, got {actual}"
                )

    unexpected: list[str] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root).as_posix()
        if relative_dir != ".":
            if not any(
                relative_dir == node or relative_dir.startswith(f"{node}/") or node.startswith(f"{relative_dir}/")
                for node in nodes
            ):
                unexpected.append(relative_dir)
        for name in directories:
            relative = (current_path / name).relative_to(root).as_posix()
            if not any(
                relative == node or relative.startswith(f"{node}/") or node.startswith(f"{relative}/")
                for node in nodes
            ):
                unexpected.append(relative)
        for name in files:
            relative = (current_path / name).relative_to(root).as_posix()
            if relative == "overlay-manifest.yaml":
                continue
            if not any(relative == node or relative.startswith(f"{node}/") for node in nodes):
                unexpected.append(relative)
    if unexpected:
        raise ReleaseArtifactError(
            "overlay contains paths outside declared nodes: " + ", ".join(sorted(set(unexpected))[:20])
        )


def _iter_overlay_files(overlay: Path) -> Iterable[tuple[str, Path]]:
    manifest_path = overlay / "overlay-manifest.yaml"
    yield "overlay/overlay-manifest.yaml", manifest_path
    for path in sorted(overlay.rglob("*"), key=lambda item: item.as_posix()):
        if path == manifest_path or not (path.is_file() or path.is_symlink()):
            continue
        if any(part in INTERNAL_TOOL_DIRS for part in path.relative_to(overlay).parts):
            continue
        yield f"overlay/{path.relative_to(overlay).as_posix()}", path


def tracked_data_files(repo: Path) -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseArtifactError(f"cannot list tracked data files in {repo}: {exc}") from exc
    files: list[Path] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8", errors="surrogateescape"))
        if "data" not in relative.parts:
            continue
        path = repo / relative
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(repo).as_posix())


def _tar_info(arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _build_tar_gz(
    destination: Path,
    entries: Iterable[tuple[str, Path]],
    *,
    directories: Iterable[str] = (),
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    ordered = sorted(entries, key=lambda item: item[0])
    directory_entries: list[str] = []
    for value in directories:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ReleaseArtifactError(f"invalid release directory entry: {value!r}")
        directory_entries.append(relative.as_posix())
    try:
        with open(temporary, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for arcname in sorted(set(directory_entries)):
                        info = _tar_info(arcname)
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        archive.addfile(info)
                    for arcname, source in ordered:
                        info = _tar_info(arcname)
                        if source.is_symlink():
                            target = os.readlink(source)
                            if Path(target).is_absolute() or ".." in Path(target).parts:
                                raise ReleaseArtifactError(f"unsafe symlink in release input: {source} -> {target}")
                            info.type = tarfile.SYMTYPE
                            info.mode = 0o777
                            info.linkname = Path(target).as_posix()
                            archive.addfile(info)
                        else:
                            payload = canonical_bytes(source.read_bytes())
                            info.type = tarfile.REGTYPE
                            info.mode = 0o644
                            info.size = len(payload)
                            archive.addfile(info, io.BytesIO(payload))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_artifacts(
    *,
    repo: Path,
    overlay: Path,
    output: Path,
    data_files: list[Path] | None = None,
    agent_gym_wheel: Path | None = None,
    framework_wheel: Path | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    overlay = overlay.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(overlay)
    validate_overlay_tree(overlay, manifest)

    version = str(manifest.get("overlay_version") or "").strip()
    if not version:
        raise ReleaseArtifactError("overlay_version is required")
    safe_version = "".join(character if character.isalnum() or character in ".-_" else "-" for character in version)
    overlay_name = f"agent-gym-overlay-{safe_version}.tar.gz"
    data_name = f"agent-gym-data-{safe_version}.tar.gz"

    overlay_entries = list(_iter_overlay_files(overlay))
    _build_tar_gz(output / overlay_name, overlay_entries)

    files = data_files if data_files is not None else tracked_data_files(repo)
    data_entries: list[tuple[str, Path]] = []
    for path in files:
        path = path.resolve()
        try:
            relative = path.relative_to(repo).as_posix()
        except ValueError as exc:
            raise ReleaseArtifactError(f"data file is outside repo: {path}") from exc
        data_entries.append((f"data/{relative}", path))
    _build_tar_gz(output / data_name, data_entries, directories=("data",))

    artifacts: dict[str, Any] = {}
    for key, name, count in (
        ("overlay", overlay_name, len(overlay_entries)),
        ("data", data_name, len(data_entries)),
    ):
        path = output / name
        artifacts[key] = {
            "file": name,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "entries": count,
        }

    release: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "overlay_version": version,
        "repo": manifest.get("repo") or {},
        "framework": manifest.get("framework") or {},
        "artifacts": artifacts,
        "install": {
            "overlay_dir": "overlay",
            "data_dir": "data",
            "workspace_dir": "workspace",
        },
    }
    if agent_gym_wheel is not None:
        wheel = agent_gym_wheel.resolve()
        if not wheel.is_file() or wheel.suffix != ".whl":
            raise ReleaseArtifactError(f"agent-gym wheel does not exist: {wheel}")
        destination = output / wheel.name
        if destination.resolve() != wheel:
            shutil.copyfile(wheel, destination)
        release["agent_gym_wheel"] = {
            "file": wheel.name,
            "sha256": sha256_file(destination),
            "size": destination.stat().st_size,
        }
    if framework_wheel is not None:
        wheel = framework_wheel.resolve()
        if not wheel.is_file() or wheel.suffix != ".whl":
            raise ReleaseArtifactError(f"framework wheel does not exist: {wheel}")
        expected = str((manifest.get("framework") or {}).get("wheel_sha256") or "").removeprefix("sha256:").lower()
        actual = sha256_file(wheel)
        if not expected or actual != expected:
            raise ReleaseArtifactError(
                f"framework wheel sha256 does not match overlay manifest: expected {expected or '<missing>'}, got {actual}"
            )
        destination = output / wheel.name
        if destination.resolve() != wheel:
            shutil.copyfile(wheel, destination)
        release["framework_wheel"] = {
            "file": wheel.name,
            "sha256": actual,
            "size": destination.stat().st_size,
        }
    release_path = output / "release-manifest.json"
    release_path.write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for artifact in artifacts.values():
        checksum = output / f"{artifact['file']}.sha256"
        checksum.write_text(f"{artifact['sha256']}  {artifact['file']}\n", encoding="ascii")
    if "framework_wheel" in release:
        artifact = release["framework_wheel"]
        checksum = output / f"{artifact['file']}.sha256"
        checksum.write_text(f"{artifact['sha256']}  {artifact['file']}\n", encoding="ascii")
    return release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-gym-wheel", type=Path, default=None)
    parser.add_argument("--framework-wheel", type=Path, default=None)
    args = parser.parse_args(argv)
    overlay = args.overlay or args.repo / "overlay"
    release = build_artifacts(
        repo=args.repo,
        overlay=overlay,
        output=args.output,
        agent_gym_wheel=args.agent_gym_wheel,
        framework_wheel=args.framework_wheel,
    )
    print(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
