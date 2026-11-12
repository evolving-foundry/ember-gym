from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from agent_gym import overlay as overlay_module
from agent_gym.launcher import (
    _absolutize_path_arguments,
    _is_diagnostic_invocation,
    _needs_prepare_staging,
    _shadowing_component_root,
)
from agent_gym.layout import GymLayout, LayoutError
from agent_gym.overlay import OverlayError, tree_sha256, verify_manifest
from agent_gym.staging import (
    STAGE_MANIFEST,
    activate_staged_components,
    cleanup_staged_components,
    stage_prepare_components,
)


def make_layout(tmp_path: Path, *, data_root: bool = True) -> GymLayout:
    repo = tmp_path / "repo"
    overlay = repo / "overlay"
    workspace = tmp_path / "workspace"
    framework = tmp_path / "site-packages"
    data = tmp_path / "data" if data_root else None
    for path in (repo, overlay, workspace, framework):
        path.mkdir(parents=True, exist_ok=True)
    touch(framework, "nemo_gym/__init__.py")
    if data is not None:
        data.mkdir(parents=True, exist_ok=True)
    return GymLayout(
        repo_root=repo,
        overlay_root=overlay,
        workspace_root=workspace,
        data_root=data,
        extra_roots=(),
        framework_root=framework,
    )


def touch(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relative, encoding="utf-8")
    return path


def test_resolve_uses_overlay_then_repo_then_workspace_then_wheel(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    relative = "benchmarks/demo/config.yaml"
    framework_file = touch(layout.framework_root, relative)
    workspace_file = touch(layout.workspace_root, relative)
    repo_file = touch(layout.repo_root, relative)
    overlay_file = touch(layout.overlay_root, relative)

    assert layout.resolve(relative) == overlay_file.resolve()
    overlay_file.unlink()
    assert layout.resolve(relative) == repo_file.resolve()
    repo_file.unlink()
    assert layout.resolve(relative) == workspace_file.resolve()
    workspace_file.unlink()
    assert layout.resolve(relative) == framework_file.resolve()


def test_resolve_data_prefers_data_root(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    relative = "benchmarks/demo/data/train.jsonl"
    overlay_file = touch(layout.overlay_root, relative)
    data_file = touch(layout.data_root, relative)

    assert layout.resolve_data(relative) == data_file.resolve()
    data_file.unlink()
    assert layout.resolve_data(relative) == overlay_file.resolve()


def test_explicit_data_root_is_a_priority_not_an_exclusive_root(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    relative = "benchmarks/demo/data/train.jsonl"
    repo_file = touch(layout.repo_root, relative)

    assert layout.resolve_data(relative) == repo_file.resolve()


def test_discover_reads_explicit_environment_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    overlay = tmp_path / "deployment-overlay"
    workspace = tmp_path / "workspace"
    framework = tmp_path / "site-packages"
    for path in (repo, overlay, workspace, framework):
        path.mkdir(parents=True)
    monkeypatch.setenv("AGENT_GYM_OVERLAY", str(overlay))
    monkeypatch.setenv("AGENT_GYM_WORKSPACE", str(workspace))
    monkeypatch.delenv("AGENT_GYM_DATA_ROOT", raising=False)
    monkeypatch.delenv("AGENT_GYM_EXTRA_ROOTS", raising=False)
    monkeypatch.delenv("NEMO_GYM_EXTRA_ROOTS", raising=False)

    layout = GymLayout.discover(repo_root=repo, framework_root=framework)

    assert layout.overlay_root == overlay.resolve()
    assert layout.workspace_root == workspace.resolve()
    assert layout.data_root == (workspace / "data").resolve()
    assert layout.search_roots[:3] == (overlay.resolve(), repo.resolve(), workspace.resolve())


def test_overlay_only_environment_infers_repo_from_overlay_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = tmp_path / "deployment"
    overlay = deployment / "overlay"
    overlay.mkdir(parents=True)
    caller = tmp_path / "elsewhere"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setenv("AGENT_GYM_OVERLAY", str(overlay))
    for name in (
        "AGENT_GYM_REPO",
        "AGENT_GYM_ROOT",
        "AGENT_GYM_WORKSPACE",
        "AGENT_GYM_DATA_ROOT",
        "AGENT_GYM_EXTRA_ROOTS",
        "NEMO_GYM_EXTRA_ROOTS",
    ):
        monkeypatch.delenv(name, raising=False)

    layout = GymLayout.discover(framework_root=tmp_path / "site-packages")

    assert layout.overlay_root == overlay.resolve()
    assert layout.repo_root == deployment.resolve()
    assert layout.workspace_root == deployment.resolve()
    assert layout.data_root == (deployment / "data").resolve()


def test_source_repo_is_found_from_a_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "source"
    subdirectory = repo / "benchmarks" / "demo"
    subdirectory.mkdir(parents=True)
    touch(repo, "agent_gym/layout.py")
    touch(repo, "pyproject.toml")
    touch(repo, "overlay/overlay-manifest.yaml")
    monkeypatch.chdir(subdirectory)
    for name in (
        "AGENT_GYM_REPO",
        "AGENT_GYM_ROOT",
        "AGENT_GYM_OVERLAY",
        "AGENT_GYM_WORKSPACE",
        "AGENT_GYM_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    layout = GymLayout.discover(framework_root=tmp_path / "site-packages")

    assert layout.repo_root == repo.resolve()
    assert layout.workspace_root == repo.resolve()
    assert layout.overlay_root == (repo / "overlay").resolve()


def test_nested_deployment_root_wins_over_ancestor_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    deployment = source / ".tmp" / "deployment"
    touch(source, "agent_gym/layout.py")
    touch(source, "pyproject.toml")
    touch(source, "overlay/overlay-manifest.yaml")
    touch(deployment, "overlay/overlay-manifest.yaml")
    touch(deployment, "deployment.json")
    monkeypatch.chdir(deployment)
    for name in (
        "AGENT_GYM_REPO",
        "AGENT_GYM_ROOT",
        "AGENT_GYM_OVERLAY",
        "AGENT_GYM_WORKSPACE",
        "AGENT_GYM_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    layout = GymLayout.discover(framework_root=tmp_path / "site-packages")

    assert layout.repo_root == deployment.resolve()
    assert layout.workspace_root == deployment.resolve()
    assert layout.overlay_root == (deployment / "overlay").resolve()
    assert layout.data_root == (deployment / "data").resolve()


def test_gym_env_exports_overlay_before_repo(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    env = layout.gym_env(environ={})

    assert env["AGENT_GYM_OVERLAY"] == str(layout.overlay_root)
    assert env["AGENT_GYM_WORKSPACE"] == str(layout.workspace_root)
    assert env["NEMO_GYM_EXTRA_ROOTS"].split(os.pathsep) == [
        str(layout.overlay_root),
        str(layout.repo_root),
        str(layout.workspace_root),
    ]
    assert env["AGENT_GYM_DATA_ROOT"] == str(layout.data_root)


def test_resolve_write_uses_workspace_for_relative_paths(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    absolute = tmp_path / "absolute" / "results"

    assert layout.resolve_write("results") == (layout.workspace_root / "results").resolve()
    assert layout.resolve_write(absolute) == absolute.resolve()


def test_launcher_restores_caller_relative_paths(tmp_path: Path) -> None:
    caller = tmp_path / "caller"
    caller.mkdir()
    argv = [
        "gym",
        "--config",
        "configs/demo.yaml",
        "--input=input.jsonl",
        "--output",
        "nested/out.jsonl",
        "--limit",
        "3",
    ]

    rewritten = _absolutize_path_arguments(argv, caller)

    assert rewritten[1:3] == ["--config", str((caller / "configs/demo.yaml").resolve())]
    assert rewritten[3] == f"--input={(caller / 'input.jsonl').resolve()}"
    assert rewritten[5] == str((caller / "nested/out.jsonl").resolve())
    assert rewritten[-2:] == ["--limit", "3"]
    assert _is_diagnostic_invocation(["gym", "--help"])
    assert _is_diagnostic_invocation(["gym", "eval", "--version"])
    assert not _is_diagnostic_invocation(["gym", "eval", "run"])


def test_launcher_resolves_config_from_layout_when_absent_from_caller_cwd(
    tmp_path: Path,
) -> None:
    layout = make_layout(tmp_path)
    config = touch(layout.overlay_root, "benchmarks/demo/config.yaml")
    caller = tmp_path / "caller"
    caller.mkdir()
    argv = [
        "gym",
        "eval",
        "run",
        "--config",
        "benchmarks/demo/config.yaml",
        "--output",
        "results/out.jsonl",
    ]

    rewritten = _absolutize_path_arguments(argv, caller, layout=layout)

    assert rewritten[4] == str(config.resolve())
    assert rewritten[6] == str((caller / "results/out.jsonl").resolve())


def test_launcher_keeps_caller_config_when_it_exists(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    caller = tmp_path / "caller"
    caller.mkdir()
    caller_config = caller / "config.yaml"
    caller_config.write_text("caller: true\n", encoding="utf-8")
    touch(layout.overlay_root, "config.yaml")

    rewritten = _absolutize_path_arguments(["gym", "eval", "run", "--config=config.yaml"], caller, layout=layout)

    assert rewritten[-1] == f"--config={caller_config.resolve()}"


def test_launcher_stages_only_prepare_commands() -> None:
    assert _needs_prepare_staging("gym", ["gym", "eval", "prepare", "--benchmark", "demo"])
    assert _needs_prepare_staging("nemo_gym_prepare_benchmark", ["nemo_gym_prepare_benchmark", "--benchmark", "demo"])
    assert not _needs_prepare_staging("gym", ["gym", "eval", "run"])
    assert not _needs_prepare_staging("gym", ["gym", "eval", "prepare", "--help"])


def test_stage_prepare_components_layers_wheel_then_overlay_without_mutating_overlay(
    tmp_path: Path,
) -> None:
    layout = make_layout(tmp_path)
    wheel_benchmark = touch(layout.framework_root, "benchmarks/demo/prepare.py")
    wheel_common = touch(layout.framework_root, "benchmarks/demo/common.py")
    overlay_benchmark = touch(layout.overlay_root, "benchmarks/demo/prepare.py")
    touch(layout.overlay_root, "benchmarks/demo/config.yaml")
    touch(layout.overlay_root, "benchmarks/demo/data/source.jsonl")
    touch(layout.framework_root, "resources_servers/helper/app.py")
    overlay_hash = tree_sha256(layout.overlay_root / "benchmarks/demo")

    staged = stage_prepare_components(layout)

    assert (staged / "benchmarks/demo/prepare.py").read_text(encoding="utf-8") == str(
        overlay_benchmark.relative_to(layout.overlay_root).as_posix()
    )
    assert (staged / "benchmarks/demo/common.py").read_text(encoding="utf-8") == str(
        wheel_common.relative_to(layout.framework_root).as_posix()
    )
    assert (staged / "benchmarks/demo/config.yaml").is_file()
    assert not (staged / "benchmarks/demo/data/source.jsonl").exists()
    assert (staged / "resources_servers/helper/app.py").is_file()
    assert tree_sha256(layout.overlay_root / "benchmarks/demo") == overlay_hash
    assert wheel_benchmark.is_file()


def test_stage_prepare_components_preserves_workspace_data_and_removes_stale_code(
    tmp_path: Path,
) -> None:
    layout = make_layout(tmp_path)
    data = touch(layout.workspace_root, "benchmarks/demo/data/prepared.jsonl")
    touch(layout.overlay_root, "benchmarks/demo/old.py")

    stage_prepare_components(layout)
    assert (layout.workspace_root / "benchmarks/demo/old.py").is_file()
    assert data.is_file()

    (layout.overlay_root / "benchmarks/demo/old.py").unlink()
    stage_prepare_components(layout)

    assert not (layout.workspace_root / "benchmarks/demo/old.py").exists()
    assert data.is_file()
    manifest = (layout.workspace_root / STAGE_MANIFEST).read_text(encoding="utf-8")
    assert "benchmarks/demo/data" not in manifest


def test_cleanup_staged_components_preserves_generated_files_and_removes_manifest(
    tmp_path: Path,
) -> None:
    layout = make_layout(tmp_path)
    touch(layout.overlay_root, "benchmarks/demo/prepare.py")
    stage_prepare_components(layout)
    generated = touch(layout.workspace_root, "benchmarks/demo/data/prepared.jsonl")
    untracked = touch(layout.workspace_root, "benchmarks/demo/custom-notes.txt")
    cache = touch(layout.workspace_root, "benchmarks/demo/__pycache__/prepare.pyc")

    removed = cleanup_staged_components(layout)

    assert removed >= 1
    assert not (layout.workspace_root / "benchmarks/demo/prepare.py").exists()
    assert generated.is_file()
    assert untracked.is_file()
    assert not cache.exists()
    assert not (layout.workspace_root / STAGE_MANIFEST).exists()
    assert cleanup_staged_components(layout) == 0


def test_activate_staged_components_puts_workspace_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = make_layout(tmp_path)
    overlay = str(layout.overlay_root.resolve())
    workspace = str(layout.workspace_root.resolve())
    monkeypatch.setattr(sys, "path", [overlay, workspace, "/site-packages"])

    activate_staged_components(layout)

    assert sys.path == [workspace, overlay, "/site-packages"]


def test_framework_shadow_fails_fast(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    touch(layout.repo_root, "nemo_gym/__init__.py")

    with pytest.raises(LayoutError, match="framework package shadowed"):
        layout.assert_framework_not_shadowed()


def test_launcher_allows_installed_framework_below_repo_but_rejects_component_shadow(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    installed = repo / ".venv" / "Lib" / "site-packages" / "nemo_gym" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    shadow = repo / "nemo_gym" / "__init__.py"
    shadow.parent.mkdir()
    shadow.write_text("", encoding="utf-8")

    roots = (repo,)
    assert _shadowing_component_root(installed, roots) is None
    assert _shadowing_component_root(shadow, roots) == (repo / "nemo_gym").resolve()


def test_overlay_manifest_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    node = layout.overlay_root / "benchmarks/demo"
    touch(node, "config.yaml")
    manifest = {
        "schema_version": 2,
        "overlay_version": "test",
        "framework": {"distribution": "nemo-gym", "version": "0.7.0"},
        "components": {"benchmarks/demo": {"tree_sha256": tree_sha256(node)}},
        "assets": {},
    }
    layout.manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    assert verify_manifest(layout, verify_framework=False)["overlay_version"] == "test"
    (node / "config.yaml").write_text("tampered", encoding="utf-8")
    with pytest.raises(OverlayError, match="tree hash mismatch"):
        verify_manifest(layout, verify_framework=False)


def test_overlay_rejects_unmanifested_paths(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    node = layout.overlay_root / "benchmarks/demo"
    touch(node, "config.yaml")
    touch(layout.overlay_root, "benchmarks/stray/extra.py")
    manifest = {
        "schema_version": 2,
        "overlay_version": "test",
        "framework": {"distribution": "nemo-gym", "version": "0.7.0"},
        "components": {"benchmarks/demo": {"tree_sha256": tree_sha256(node)}},
        "assets": {},
        "tools": {},
    }
    layout.manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    with pytest.raises(OverlayError, match="outside declared manifest nodes"):
        verify_manifest(layout, verify_framework=False)


def test_tree_hash_is_line_ending_independent(tmp_path: Path) -> None:
    node = tmp_path / "node"
    node.mkdir()
    payload = node / "config.yaml"

    payload.write_bytes(b"key: value\nsecond: line\n")
    lf_hash = tree_sha256(node)
    payload.write_bytes(b"key: value\r\nsecond: line\r\n")

    assert tree_sha256(node) == lf_hash


def test_tree_hash_is_executable_bit_independent(tmp_path: Path) -> None:
    node = tmp_path / "node"
    node.mkdir()
    payload = node / "prepare.py"
    payload.write_bytes(b"print('ok')\n")
    baseline = tree_sha256(node)

    payload.chmod(0o755)

    assert tree_sha256(node) == baseline


class InstalledDistribution:
    def __init__(self, version: str) -> None:
        self.version = version

    def read_text(self, name: str) -> None:
        return None


class DirectURLDistribution(InstalledDistribution):
    def __init__(self, version: str, wheel_sha256: str) -> None:
        super().__init__(version)
        self.wheel_sha256 = wheel_sha256

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json":
            return None
        return json.dumps({"archive_info": {"hash": f"sha256={self.wheel_sha256}"}})


def write_framework_manifest(layout: GymLayout, wheel_sha256: str = "abc123") -> None:
    node = layout.overlay_root / "benchmarks/demo"
    touch(node, "config.yaml")
    package_tree_sha256 = tree_sha256(layout.framework_root / "nemo_gym")
    manifest = {
        "schema_version": 2,
        "overlay_version": "test",
        "framework": {
            "distribution": "nemo-gym",
            "version": "0.7.0",
            "wheel_sha256": wheel_sha256,
            "package_tree_sha256": package_tree_sha256,
        },
        "components": {"benchmarks/demo": {"tree_sha256": tree_sha256(node)}},
        "assets": {},
        "tools": {},
    }
    layout.manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")


def test_manifest_checks_installed_framework_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = make_layout(tmp_path)
    write_framework_manifest(layout)
    (layout.repo_root / "uv.lock").write_text(
        '[[package]]\nname = "nemo-gym"\nversion = "0.7.0"\nwheels = [{ hash = "sha256:abc123" }]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        overlay_module.importlib.metadata,
        "distribution",
        lambda _: InstalledDistribution("0.7.0"),
    )

    assert verify_manifest(layout)["framework"]["version"] == "0.7.0"

    monkeypatch.setattr(
        overlay_module.importlib.metadata,
        "distribution",
        lambda _: InstalledDistribution("0.6.0"),
    )
    with pytest.raises(OverlayError, match="does not match the installed nemo-gym"):
        verify_manifest(layout)


def test_manifest_checks_wheel_hash_against_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = make_layout(tmp_path)
    write_framework_manifest(layout, wheel_sha256="deadbeef")
    (layout.repo_root / "uv.lock").write_text(
        '[[package]]\nname = "nemo-gym"\nversion = "0.7.0"\nwheels = [{ hash = "sha256:abc123" }]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        overlay_module.importlib.metadata,
        "distribution",
        lambda _: InstalledDistribution("0.7.0"),
    )

    with pytest.raises(OverlayError, match="wheel_sha256 does not match"):
        verify_manifest(layout)


def test_manifest_accepts_pep610_wheel_hash_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = make_layout(tmp_path)
    write_framework_manifest(layout, wheel_sha256="abc123")
    monkeypatch.setattr(
        overlay_module.importlib.metadata,
        "distribution",
        lambda _: DirectURLDistribution("0.7.0", "abc123"),
    )

    assert verify_manifest(layout)["framework"]["wheel_sha256"] == "abc123"


def test_manifest_accepts_any_wheel_hash_from_multi_wheel_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = make_layout(tmp_path)
    write_framework_manifest(layout, wheel_sha256="abc123")
    (layout.repo_root / "uv.lock").write_text(
        '[[package]]\nname = "nemo-gym"\nversion = "0.7.0"\n'
        'wheels = [{ hash = "sha256:abc123" }, { hash = "sha256:def456" }]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        overlay_module.importlib.metadata,
        "distribution",
        lambda _: InstalledDistribution("0.7.0"),
    )

    assert verify_manifest(layout)["framework"]["wheel_sha256"] == "abc123"


def test_manifest_fails_closed_without_wheel_hash_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = make_layout(tmp_path)
    write_framework_manifest(layout)
    monkeypatch.setattr(
        overlay_module.importlib.metadata,
        "distribution",
        lambda _: InstalledDistribution("0.7.0"),
    )

    with pytest.raises(OverlayError, match="no wheel hash evidence"):
        verify_manifest(layout)


def test_manifest_verifies_package_tree_even_with_explicit_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = make_layout(tmp_path)
    write_framework_manifest(layout)
    wheel = tmp_path / "nemo_gym-0.7.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    monkeypatch.setenv("AGENT_GYM_NEMO_WHEEL", str(wheel))
    monkeypatch.setattr(
        overlay_module.importlib.metadata,
        "distribution",
        lambda _: InstalledDistribution("0.7.0"),
    )

    with pytest.raises(OverlayError, match="wheel_sha256 does not match"):
        verify_manifest(layout)

    write_framework_manifest(layout, wheel_sha256=overlay_module.sha256_file(wheel))
    assert verify_manifest(layout)["framework"]["version"] == "0.7.0"
    touch(layout.framework_root, "nemo_gym/unexpected.py")
    with pytest.raises(OverlayError, match="package_tree_sha256 does not match"):
        verify_manifest(layout)
