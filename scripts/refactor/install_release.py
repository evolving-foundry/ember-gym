"""Install and verify AgentGym overlay/data release artifacts outside the source checkout."""

from __future__ import annotations

import argparse
import email.parser
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


try:
    from build_release_artifacts import sha256_file
except ImportError:  # pragma: no cover - only used when imported through importlib specs
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_release_artifacts import sha256_file


class ReleaseInstallError(RuntimeError):
    """Raised when a release artifact cannot be installed safely."""


def load_release_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseInstallError(f"release manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseInstallError(f"cannot read release manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ReleaseInstallError("release manifest schema_version must be 1")
    return manifest


def verify_artifact(root: Path, artifact: dict[str, Any]) -> Path:
    name = artifact.get("file")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ReleaseInstallError(f"invalid artifact file name: {name!r}")
    path = root / name
    if not path.is_file():
        raise ReleaseInstallError(f"artifact does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != artifact.get("size"):
        raise ReleaseInstallError(
            f"artifact size mismatch for {name}: expected {artifact.get('size')}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != artifact.get("sha256"):
        raise ReleaseInstallError(
            f"artifact sha256 mismatch for {name}: expected {artifact.get('sha256')}, got {actual_hash}"
        )
    return path


def _validate_members(archive: Path, expected_top: str) -> None:
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ReleaseInstallError(f"unsafe archive member in {archive.name}: {member.name}")
            if name.parts[0] != expected_top:
                raise ReleaseInstallError(f"archive {archive.name} contains unexpected top-level entry: {member.name}")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or ".." in target.parts:
                    raise ReleaseInstallError(f"unsafe link in {archive.name}: {member.name} -> {member.linkname}")
            elif not (member.isfile() or member.isdir()):
                raise ReleaseInstallError(f"unsupported archive member type in {archive.name}: {member.name}")


def _extract(archive: Path, destination: Path, expected_top: str) -> None:
    _validate_members(archive, expected_top)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as handle:
        handle.extractall(destination, filter="data")


def wheel_install_commands(
    python: Path, framework_wheel: Path, agent_gym_wheel: Path, *, install_deps: bool = False
) -> list[list[str]]:
    """Return the only supported release install order: framework first, AgentGym last."""

    def command(wheel: Path) -> list[str]:
        result = [str(python), "-m", "pip", "install", "--force-reinstall"]
        if not install_deps:
            result.extend(["--no-deps", "--no-index"])
        result.append(str(wheel))
        return result

    return [command(framework_wheel), command(agent_gym_wheel)]


def installed_release_check_command(
    python: Path,
    expected_distributions: dict[str, str],
    expected_modules: dict[str, list[str]] | None = None,
) -> list[str]:
    """Verify installed wheels without importing from the source checkout."""

    script = (
        "import importlib.metadata as m, importlib.util as u, json, pathlib, sys\n"
        "expected = json.loads(sys.argv[1])\n"
        "modules = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}\n"
        "actual = {name: m.version(name) for name in expected}\n"
        "locations = {name: str(m.distribution(name).locate_file('')) for name in expected}\n"
        "errors = []\n"
        "if actual != expected:\n"
        "    errors.append(f'distribution versions mismatch: {actual!r}')\n"
        "for distribution, names in modules.items():\n"
        "    root = pathlib.Path(locations[distribution]).resolve()\n"
        "    for name in names:\n"
        "        spec = u.find_spec(name)\n"
        "        if spec is None:\n"
        "            errors.append(f'module is not importable: {name}')\n"
        "            continue\n"
        "        origins = list(spec.submodule_search_locations or ()) or ([spec.origin] if spec.origin else [])\n"
        "        if not origins or any(not pathlib.Path(origin).resolve().is_relative_to(root) for origin in origins):\n"
        "            errors.append(f'module {name} is not installed under {root}: {origins!r}')\n"
        "print(json.dumps({'versions': actual, 'locations': locations, 'errors': errors}, sort_keys=True))\n"
        "raise SystemExit(0 if not errors else 1)\n"
    )
    command = [
        str(python),
        "-I",
        "-c",
        script,
        json.dumps(expected_distributions, sort_keys=True),
    ]
    if expected_modules is not None:
        command.append(json.dumps(expected_modules, sort_keys=True))
    return command


def wheel_distribution_identity(wheel: Path) -> tuple[str, str]:
    """Read the canonical distribution name and version from wheel METADATA."""

    with zipfile.ZipFile(wheel) as archive:
        metadata_members = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        ]
        if len(metadata_members) != 1:
            raise ReleaseInstallError(f"wheel must contain exactly one .dist-info/METADATA: {wheel}")
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_members[0]))
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ReleaseInstallError(f"wheel METADATA is missing Name/Version: {wheel}")
    return name, version


def wheel_import_packages(wheel: Path) -> list[str]:
    """Return top-level import packages provided by a wheel."""

    packages: set[str] = set()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if not parts or parts[0].endswith((".dist-info", ".data")):
                continue
            if len(parts) == 1 and parts[0].endswith(".py"):
                packages.add(PurePosixPath(parts[0]).stem)
            elif len(parts) >= 2 and parts[1] == "__init__.py":
                packages.add(parts[0])
    return sorted(packages)


def _install_wheels(
    python: Path,
    framework_wheel: Path,
    agent_gym_wheel: Path,
    *,
    install_deps: bool,
    strict_pip_check: bool,
) -> None:
    commands = wheel_install_commands(
        python,
        framework_wheel,
        agent_gym_wheel,
        install_deps=install_deps,
    )
    for command in commands:
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReleaseInstallError(f"cannot install release wheel: {exc}") from exc
    expected = dict(wheel_distribution_identity(wheel) for wheel in (framework_wheel, agent_gym_wheel))
    expected_modules = {
        distribution: wheel_import_packages(wheel)
        for wheel in (framework_wheel, agent_gym_wheel)
        for distribution in (wheel_distribution_identity(wheel)[0],)
    }
    try:
        subprocess.run(
            installed_release_check_command(python, expected, expected_modules),
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseInstallError(f"installed release distributions do not match the wheels: {exc}") from exc
    if strict_pip_check:
        try:
            subprocess.run([str(python), "-m", "pip", "check"], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ReleaseInstallError(f"installed release failed pip check: {exc}") from exc


def install_release(
    *,
    release_manifest: Path,
    prefix: Path,
    workspace: Path | None = None,
    python: Path | None = None,
    install_deps: bool = False,
    strict_pip_check: bool = False,
) -> dict[str, Any]:
    manifest = load_release_manifest(release_manifest)
    prefix = prefix.resolve(strict=False)
    overlay_target = prefix / "overlay"
    data_target = prefix / "data"
    framework_target = prefix / "framework"
    agent_gym_target = prefix / "agent-gym"
    if overlay_target.exists() or data_target.exists() or framework_target.exists() or agent_gym_target.exists():
        raise ReleaseInstallError(
            f"install target already contains release roots: {prefix}; install into a clean prefix"
        )

    artifacts = manifest.get("artifacts") or {}
    overlay_archive = verify_artifact(release_manifest.parent, artifacts.get("overlay") or {})
    data_archive = verify_artifact(release_manifest.parent, artifacts.get("data") or {})
    framework_artifact = manifest.get("framework_wheel")
    framework_source = verify_artifact(release_manifest.parent, framework_artifact) if framework_artifact else None
    agent_gym_artifact = manifest.get("agent_gym_wheel")
    agent_gym_source = verify_artifact(release_manifest.parent, agent_gym_artifact) if agent_gym_artifact else None
    if python is not None and (framework_source is None or agent_gym_source is None):
        raise ReleaseInstallError("installing release wheels requires both framework_wheel and agent_gym_wheel artifacts")

    prefix.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{prefix.name}.install-", dir=prefix.parent))
    try:
        _extract(overlay_archive, staging, "overlay")
        _extract(data_archive, staging, "data")
        shutil.move(str(staging / "overlay"), str(overlay_target))
        shutil.move(str(staging / "data"), str(data_target))
    except (OSError, tarfile.TarError) as exc:
        if overlay_target.exists():
            shutil.rmtree(overlay_target, ignore_errors=True)
        if data_target.exists():
            shutil.rmtree(data_target, ignore_errors=True)
        raise ReleaseInstallError(f"cannot extract release artifacts: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    workspace_root = (workspace or prefix / "workspace").resolve(strict=False)
    workspace_root.mkdir(parents=True, exist_ok=True)
    framework_path = None
    if framework_source is not None:
        framework_target.mkdir(parents=True, exist_ok=True)
        framework_path = framework_target / framework_source.name
        shutil.copyfile(framework_source, framework_path)
    agent_gym_path = None
    if agent_gym_source is not None:
        agent_gym_target.mkdir(parents=True, exist_ok=True)
        agent_gym_path = agent_gym_target / agent_gym_source.name
        shutil.copyfile(agent_gym_source, agent_gym_path)
    if python is not None:
        assert framework_path is not None and agent_gym_path is not None
        _install_wheels(
            python,
            framework_path,
            agent_gym_path,
            install_deps=install_deps,
            strict_pip_check=strict_pip_check,
        )
    env = {
        "AGENT_GYM_ROOT": str(prefix),
        "AGENT_GYM_REPO": str(prefix),
        "AGENT_GYM_OVERLAY": str(overlay_target),
        "AGENT_GYM_DATA_ROOT": str(data_target),
        "AGENT_GYM_WORKSPACE": str(workspace_root),
    }
    if framework_path is not None:
        env["AGENT_GYM_NEMO_WHEEL"] = str(framework_path)
    deployment = {
        "schema_version": 1,
        "overlay_version": manifest.get("overlay_version"),
        "roots": {
            "repo": str(prefix),
            "overlay": str(overlay_target),
            "data": str(data_target),
            "workspace": str(workspace_root),
        },
        "env": env,
        "artifacts": artifacts,
        "agent_gym_wheel": manifest.get("agent_gym_wheel"),
        "framework_wheel": manifest.get("framework_wheel"),
        "wheels": {
            "framework": str(framework_path) if framework_path is not None else None,
            "agent-gym": str(agent_gym_path) if agent_gym_path is not None else None,
        },
        "installed_python": str(python) if python is not None else None,
    }
    (prefix / "deployment.json").write_text(
        json.dumps(deployment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return deployment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Install the release wheels into this interpreter (framework first, AgentGym last).",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Allow pip to resolve wheel dependencies; default is an offline --no-deps install.",
    )
    parser.add_argument(
        "--strict-pip-check",
        action="store_true",
        help=(
            "Fail on global pip check errors. Disabled by default because this project "
            "intentionally excludes selected transitive dependencies."
        ),
    )
    args = parser.parse_args(argv)
    deployment = install_release(
        release_manifest=args.release_manifest,
        prefix=args.prefix,
        workspace=args.workspace,
        python=args.python,
        install_deps=args.install_deps,
        strict_pip_check=args.strict_pip_check,
    )
    print(json.dumps(deployment, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
