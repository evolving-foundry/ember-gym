from __future__ import annotations

import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from agent_gym.layout import GymLayout
from agent_gym.overlay import tree_sha256


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "refactor"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_release_artifacts import (  # noqa: E402
    ReleaseArtifactError,
    build_artifacts,
)
from install_release import (  # noqa: E402
    ReleaseInstallError,
    install_release,
    installed_release_check_command,
    wheel_distribution_identity,
    wheel_import_packages,
    wheel_install_commands,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_release_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "source"
    overlay = repo / "overlay"
    node = overlay / "benchmarks" / "demo"
    _write(node / "config.yaml", "name: demo\n")
    manifest = {
        "schema_version": 2,
        "overlay_version": "2026.10.28-test",
        "framework": {
            "distribution": "nemo-gym",
            "version": "0.7.0",
            "wheel_sha256": "a" * 64,
        },
        "repo": {"commit": "deadbeef"},
        "components": {
            "benchmarks/demo": {"tree_sha256": tree_sha256(node)},
        },
        "assets": {},
        "tools": {},
    }
    _write(overlay / "overlay-manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
    data = _write(repo / "benchmarks" / "demo" / "data" / "train.jsonl", "{}\n")
    return repo, overlay, data


def test_release_artifacts_are_deterministic_and_installable(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    first = tmp_path / "dist-a"
    second = tmp_path / "dist-b"

    release_a = build_artifacts(repo=repo, overlay=overlay, output=first, data_files=[data])
    release_b = build_artifacts(repo=repo, overlay=overlay, output=second, data_files=[data])

    assert release_a == release_b
    for artifact in release_a["artifacts"].values():
        name = artifact["file"]
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / f"{name}.sha256").read_text(encoding="ascii") == (second / f"{name}.sha256").read_text(
            encoding="ascii"
        )

    prefix = tmp_path / "deployment"
    deployment = install_release(
        release_manifest=first / "release-manifest.json",
        prefix=prefix,
    )
    assert (prefix / "overlay" / "overlay-manifest.yaml").is_file()
    assert (prefix / "data" / "benchmarks" / "demo" / "data" / "train.jsonl").is_file()
    assert deployment["env"]["AGENT_GYM_OVERLAY"] == str(prefix / "overlay")
    assert deployment["env"]["AGENT_GYM_DATA_ROOT"] == str(prefix / "data")

    layout = GymLayout.discover(
        overlay_root=prefix / "overlay",
        workspace_root=Path(deployment["env"]["AGENT_GYM_WORKSPACE"]),
        data_root=Path(deployment["env"]["AGENT_GYM_DATA_ROOT"]),
        framework_root=tmp_path / "site-packages",
    )
    assert layout.repo_root == prefix.resolve()
    assert (
        layout.resolve_data("benchmarks/demo/data/train.jsonl")
        == (prefix / "data" / "benchmarks" / "demo" / "data" / "train.jsonl").resolve()
    )


def test_empty_data_release_installs_with_empty_data_root(tmp_path: Path) -> None:
    repo, overlay, _ = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"

    release = build_artifacts(repo=repo, overlay=overlay, output=output, data_files=[])

    data_artifact = release["artifacts"]["data"]
    assert data_artifact["entries"] == 0
    with tarfile.open(output / data_artifact["file"], mode="r:gz") as archive:
        assert [(member.name, member.isdir()) for member in archive.getmembers()] == [("data", True)]

    prefix = tmp_path / "deployment"
    deployment = install_release(
        release_manifest=output / "release-manifest.json",
        prefix=prefix,
    )

    assert (prefix / "data").is_dir()
    assert list((prefix / "data").iterdir()) == []
    assert deployment["roots"]["data"] == str(prefix / "data")
    assert deployment["env"]["AGENT_GYM_DATA_ROOT"] == str(prefix / "data")


def test_release_build_rejects_unmanifested_overlay_files(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    _write(overlay / "benchmarks" / "stray" / "extra.py", "print('no')\n")

    with pytest.raises(ReleaseArtifactError, match="outside declared nodes"):
        build_artifacts(repo=repo, overlay=overlay, output=tmp_path / "dist", data_files=[data])


def test_release_excludes_runtime_metadata_from_overlay(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    node = overlay / "benchmarks" / "demo"
    _write(node / "__pycache__" / "demo.pyc", "cache")
    _write(node / "results" / "session-1" / "generation.json", "{}")
    _write(node / "data" / "runtime.jsonl", "{}\n")
    _write(node / ".tau2_data.lockdir" / "owner", "owner")

    output = tmp_path / "dist"
    release = build_artifacts(repo=repo, overlay=overlay, output=output, data_files=[data])

    with tarfile.open(output / release["artifacts"]["overlay"]["file"], mode="r:gz") as archive:
        names = archive.getnames()
    for marker in ("__pycache__", "/results/", "/data/runtime.jsonl", ".tau2_data.lockdir"):
        assert not any(marker in name for name in names), marker


def test_install_rejects_tampered_artifact(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    release = build_artifacts(repo=repo, overlay=overlay, output=output, data_files=[data])
    overlay_path = output / release["artifacts"]["overlay"]["file"]
    overlay_path.write_bytes(overlay_path.read_bytes() + b"tamper")

    with pytest.raises(ReleaseInstallError, match="size mismatch"):
        install_release(release_manifest=output / "release-manifest.json", prefix=tmp_path / "prefix")


def test_release_manifest_contains_artifact_checksums(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    build_artifacts(repo=repo, overlay=overlay, output=output, data_files=[data])
    manifest = json.loads((output / "release-manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert set(manifest["artifacts"]) == {"overlay", "data"}
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"].values())


def test_framework_wheel_is_verified_installed_and_exported(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    framework_wheel = tmp_path / "nemo_gym-0.7.0-py3-none-any.whl"
    framework_wheel.write_bytes(b"framework wheel")
    framework_sha = __import__("hashlib").sha256(framework_wheel.read_bytes()).hexdigest()
    manifest_path = overlay / "overlay-manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["framework"]["wheel_sha256"] = framework_sha
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    output = tmp_path / "dist"
    release = build_artifacts(
        repo=repo,
        overlay=overlay,
        output=output,
        data_files=[data],
        framework_wheel=framework_wheel,
    )
    assert release["framework_wheel"]["sha256"] == framework_sha
    deployment = install_release(
        release_manifest=output / "release-manifest.json",
        prefix=tmp_path / "prefix",
    )

    installed = tmp_path / "prefix" / "framework" / framework_wheel.name
    assert installed.read_bytes() == framework_wheel.read_bytes()
    assert deployment["env"]["AGENT_GYM_NEMO_WHEEL"] == str(installed)


def test_framework_wheel_must_match_manifest_hash(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    framework_wheel = tmp_path / "nemo_gym-0.7.0-py3-none-any.whl"
    framework_wheel.write_bytes(b"wrong framework wheel")

    with pytest.raises(ReleaseArtifactError, match="framework wheel sha256 does not match"):
        build_artifacts(
            repo=repo,
            overlay=overlay,
            output=tmp_path / "dist",
            data_files=[data],
            framework_wheel=framework_wheel,
        )


def test_agent_gym_wheel_is_copied_into_release_and_deployment(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    agent_gym_wheel = tmp_path / "agent_gym-0.1.0-py3-none-any.whl"
    agent_gym_wheel.write_bytes(b"agent-gym wheel")
    output = tmp_path / "dist"

    release = build_artifacts(
        repo=repo,
        overlay=overlay,
        output=output,
        data_files=[data],
        agent_gym_wheel=agent_gym_wheel,
    )

    released = output / agent_gym_wheel.name
    assert released.read_bytes() == agent_gym_wheel.read_bytes()
    assert release["agent_gym_wheel"]["sha256"] == __import__("hashlib").sha256(released.read_bytes()).hexdigest()

    deployment = install_release(
        release_manifest=output / "release-manifest.json",
        prefix=tmp_path / "prefix",
    )
    installed = tmp_path / "prefix" / "agent-gym" / agent_gym_wheel.name
    assert installed.read_bytes() == agent_gym_wheel.read_bytes()
    assert deployment["wheels"]["agent-gym"] == str(installed)


def test_release_accepts_wheels_built_directly_in_output(tmp_path: Path) -> None:
    repo, overlay, data = _make_release_inputs(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    framework_wheel = output / "nemo_gym-0.7.0-py3-none-any.whl"
    framework_wheel.write_bytes(b"framework wheel")
    framework_sha = __import__("hashlib").sha256(framework_wheel.read_bytes()).hexdigest()
    manifest_path = overlay / "overlay-manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["framework"]["wheel_sha256"] = framework_sha
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    agent_gym_wheel = output / "agent_gym-0.1.0-py3-none-any.whl"
    agent_gym_wheel.write_bytes(b"agent-gym wheel")

    release = build_artifacts(
        repo=repo,
        overlay=overlay,
        output=output,
        data_files=[data],
        framework_wheel=framework_wheel,
        agent_gym_wheel=agent_gym_wheel,
    )

    assert release["framework_wheel"]["file"] == framework_wheel.name
    assert release["agent_gym_wheel"]["file"] == agent_gym_wheel.name
    assert agent_gym_wheel.read_bytes() == b"agent-gym wheel"


def test_release_wheel_install_order_is_framework_then_agent_gym(tmp_path: Path) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    framework = tmp_path / "nemo_gym-0.7.0-py3-none-any.whl"
    agent_gym = tmp_path / "agent_gym-0.1.0-py3-none-any.whl"

    commands = wheel_install_commands(python, framework, agent_gym)

    assert [command[-1] for command in commands] == [str(framework), str(agent_gym)]
    assert all(command[:5] == [str(python), "-m", "pip", "install", "--force-reinstall"] for command in commands)
    assert all("--no-deps" in command and "--no-index" in command for command in commands)


def test_release_installation_checks_wheels_in_isolated_interpreter(tmp_path: Path) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    expected = {"agent-gym": "0.1.0", "nemo-gym": "0.7.0"}

    command = installed_release_check_command(python, expected)

    assert command[:3] == [str(python), "-I", "-c"]
    assert json.loads(command[-1]) == expected
    assert "importlib.metadata" in command[3]


def test_wheel_distribution_identity_reads_canonical_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "demo-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "demo-1.2.3.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: demo\nVersion: 1.2.3\n",
        )

    assert wheel_distribution_identity(wheel) == ("demo", "1.2.3")


def test_wheel_import_packages_lists_top_level_imports(tmp_path: Path) -> None:
    wheel = tmp_path / "demo-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo/__init__.py", "")
        archive.writestr("demo/submodule.py", "")
        archive.writestr("single_module.py", "")
        archive.writestr("demo-1.2.3.dist-info/METADATA", "Name: demo\nVersion: 1.2.3\n")

    assert wheel_import_packages(wheel) == ["demo", "single_module"]


def test_release_installation_checks_module_provenance() -> None:
    python = Path("/venv/bin/python")
    expected = {"agent-gym": "0.1.0", "nemo-gym": "0.7.0"}
    modules = {"agent-gym": ["agent_gym", "eval_service"], "nemo-gym": ["nemo_gym"]}

    command = installed_release_check_command(python, expected, modules)

    assert command[:3] == [str(python), "-I", "-c"]
    assert json.loads(command[-2]) == expected
    assert json.loads(command[-1]) == modules
    assert "find_spec" in command[3]
    assert "root = pathlib.Path(locations[distribution]).resolve()" in command[3]
