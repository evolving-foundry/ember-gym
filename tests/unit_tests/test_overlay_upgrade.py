import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def _load_audit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "refactor" / "audit_tree.py"
    spec = importlib.util.spec_from_file_location("agent_gym_audit_tree", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "refactor" / "stage_upgrade_tree.py"
    spec = importlib.util.spec_from_file_location("agent_gym_stage_upgrade_tree", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest_update_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "refactor" / "update_overlay_manifest.py"
    spec = importlib.util.spec_from_file_location("agent_gym_update_overlay_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_wheel_mode_accepts_new_upstream_when_overlay_did_not_touch_file(tmp_path):
    audit = _load_audit_module()
    old = tmp_path / "old"
    new = tmp_path / "new"
    local = tmp_path / "local"
    local_file = _write(local, "component/app.py", "old\n")
    old_file = _write(old, "component/app.py", "old\n")
    new_file = _write(new, "component/app.py", "new\n")

    selected, conflicts, states = audit.plan_node(
        "responses_api_agents/sample",
        {"app.py": local_file},
        {"app.py": old_file},
        {"app.py": new_file},
        baseline_mode="wheel",
    )

    assert selected["app.py"] == new_file
    assert conflicts == []
    assert states["upstream_unchanged"] == 1


def test_wheel_mode_preserves_overlay_change_when_upstream_is_unchanged(tmp_path):
    audit = _load_audit_module()
    old = tmp_path / "old"
    new = tmp_path / "new"
    local = tmp_path / "local"
    local_file = _write(local, "component/app.py", "custom\n")
    old_file = _write(old, "component/app.py", "old\n")
    new_file = _write(new, "component/app.py", "old\n")

    selected, conflicts, states = audit.plan_node(
        "responses_api_agents/sample",
        {"app.py": local_file},
        {"app.py": old_file},
        {"app.py": new_file},
        baseline_mode="wheel",
    )

    assert selected["app.py"] == local_file
    assert conflicts == []
    assert states["local_changed"] == 1


def test_wheel_mode_reports_three_way_conflict_and_keeps_overlay(tmp_path):
    audit = _load_audit_module()
    old = tmp_path / "old"
    new = tmp_path / "new"
    local = tmp_path / "local"
    local_file = _write(local, "component/app.py", "custom\n")
    old_file = _write(old, "component/app.py", "old\n")
    new_file = _write(new, "component/app.py", "new\n")

    selected, conflicts, states = audit.plan_node(
        "responses_api_agents/sample",
        {"app.py": local_file},
        {"app.py": old_file},
        {"app.py": new_file},
        baseline_mode="wheel",
    )

    assert selected["app.py"] == local_file
    assert states["conflict"] == 1
    assert conflicts == [
        {
            "node": "responses_api_agents/sample",
            "path": "app.py",
            "state": "both_changed_different",
            "resolution": "keep_local",
        }
    ]


def test_wheel_mode_accepts_upstream_removal_when_overlay_did_not_touch_file(tmp_path):
    audit = _load_audit_module()
    old = tmp_path / "old"
    new = tmp_path / "new"
    local = tmp_path / "local"
    local_file = _write(local, "component/app.py", "old\n")
    old_file = _write(old, "component/app.py", "old\n")
    new.mkdir()

    selected, conflicts, states = audit.plan_node(
        "responses_api_agents/sample",
        {"app.py": local_file},
        {"app.py": old_file},
        {},
        baseline_mode="wheel",
    )

    assert selected == {}
    assert conflicts == []
    assert states["upstream_removed"] == 1


def test_plan_node_excludes_component_data_from_overlay(tmp_path):
    audit = _load_audit_module()
    local = tmp_path / "local"
    source = _write(local, "component/app.py", "custom\n")
    data = _write(local, "component/data/example.jsonl", "{}\n")

    selected, _, _ = audit.plan_node(
        "benchmarks/sample",
        {"app.py": source, "data/example.jsonl": data},
        {},
        {},
    )

    assert selected == {"app.py": source}
    assert audit.is_node_data_path("data/example.jsonl")
    assert not audit.is_node_data_path("app.py")


def test_prune_materialized_removes_stale_caches(tmp_path):
    audit = _load_audit_module()
    stale = _write(tmp_path, "component/__pycache__/app.pyc", "cache")
    wanted = _write(tmp_path, "component/app.py", "code")
    manifest = _write(tmp_path, "overlay-manifest.yaml", "manifest")

    audit.prune_materialized(tmp_path, {"component/app.py"})

    assert wanted.is_file()
    assert manifest.is_file()
    assert not stale.exists()


def test_stage_upgrade_tree_applies_overlay_over_old_wheel(tmp_path):
    stage = _load_stage_module()
    old = tmp_path / "old"
    overlay = tmp_path / "overlay"
    destination = tmp_path / "merged"
    _write(old, "component/app.py", "old\n")
    _write(old, "component/upstream.py", "upstream\n")
    _write(old, "nemo_gym/__init__.py", "wheel\n")
    _write(overlay, "component/app.py", "agent-gym\n")
    _write(overlay, "overlay-manifest.yaml", "manifest\n")

    stage.stage(old, overlay, destination)

    assert (destination / "component/app.py").read_text(encoding="utf-8") == "agent-gym\n"
    assert (destination / "component/upstream.py").read_text(encoding="utf-8") == "upstream\n"
    assert (destination / "nemo_gym/__init__.py").read_text(encoding="utf-8") == "wheel\n"
    assert not (destination / "overlay-manifest.yaml").exists()


def test_materialized_upgrade_manifest_records_target_package_tree(tmp_path, monkeypatch):
    audit = _load_audit_module()
    old = tmp_path / "old"
    new = tmp_path / "new"
    merged = tmp_path / "merged"
    overlay_next = tmp_path / "overlay-next"
    _write(old, "responses_api_agents/sample/app.py", "old\n")
    _write(new, "responses_api_agents/sample/app.py", "old\n")
    _write(new, "nemo_gym/__init__.py", "framework\n")
    _write(merged, "responses_api_agents/sample/app.py", "agent-gym\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_tree.py",
            "--repo",
            str(merged),
            "--fork-point",
            str(old),
            "--wheel",
            str(new),
            "--baseline-mode",
            "wheel",
            "--output",
            str(tmp_path / "classification.json"),
            "--graph-output",
            str(tmp_path / "graph.json"),
            "--materialize-overlay",
            str(overlay_next),
            "--manifest",
            str(overlay_next / "overlay-manifest.yaml"),
            "--framework-version",
            "0.8.0",
            "--wheel-sha256",
            "abc123",
            "--overlay-version",
            "test",
        ],
    )

    assert audit.main() == 0
    manifest = yaml.safe_load((overlay_next / "overlay-manifest.yaml").read_text(encoding="utf-8"))

    assert manifest["framework"]["package_tree_sha256"] == audit.tree_sha256(new / "nemo_gym")
    assert (overlay_next / "responses_api_agents/sample/app.py").read_text(encoding="utf-8") == "agent-gym\n"


def test_materialized_upgrade_accepts_removed_untouched_node(tmp_path, monkeypatch):
    audit = _load_audit_module()
    old = tmp_path / "old"
    new = tmp_path / "new"
    merged = tmp_path / "merged"
    overlay_next = tmp_path / "overlay-next"
    _write(old, "responses_api_agents/removed/app.py", "old\n")
    _write(merged, "responses_api_agents/removed/app.py", "old\n")
    _write(new, "responses_api_agents/kept/app.py", "kept\n")
    _write(new, "nemo_gym/__init__.py", "framework\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_tree.py",
            "--repo",
            str(merged),
            "--fork-point",
            str(old),
            "--wheel",
            str(new),
            "--baseline-mode",
            "wheel",
            "--output",
            str(tmp_path / "classification.json"),
            "--graph-output",
            str(tmp_path / "graph.json"),
            "--materialize-overlay",
            str(overlay_next),
            "--manifest",
            str(overlay_next / "overlay-manifest.yaml"),
            "--framework-version",
            "0.8.0",
            "--wheel-sha256",
            "abc123",
            "--overlay-version",
            "test",
        ],
    )

    assert audit.main() == 0
    manifest = yaml.safe_load((overlay_next / "overlay-manifest.yaml").read_text(encoding="utf-8"))

    assert "responses_api_agents/removed" not in manifest["components"]
    assert not (overlay_next / "responses_api_agents/removed").exists()


def test_tracked_files_falls_back_to_tree_walk_outside_git_root(tmp_path):
    audit = _load_audit_module()
    _write(tmp_path, "benchmarks/demo/config.yaml", "x\n")
    _write(tmp_path, "notes.txt", "x\n")

    assert audit.tracked_files(tmp_path) == ["benchmarks/demo/config.yaml", "notes.txt"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("benchmarks/sample/data/example.jsonl", True),
        ("benchmarks/sample/README.md", False),
        ("benchmarks/sample/nested/data/file.json", True),
    ],
)
def test_repo_relative_data_detection(path, expected):
    audit = _load_audit_module()
    assert audit.is_external_data_path(path) is expected


def test_update_overlay_manifest_recomputes_hashes_and_version(tmp_path):
    update = _load_manifest_update_module()
    audit = _load_audit_module()
    overlay = tmp_path / "overlay"
    node = overlay / "benchmarks/demo"
    _write(node, "config.yaml", "name: demo\n")
    manifest = {
        "schema_version": 2,
        "overlay_version": "old",
        "framework": {"distribution": "nemo-gym", "version": "0.7.0"},
        "components": {"benchmarks/demo": {"tree_sha256": "stale"}},
        "assets": {},
        "tools": {},
    }
    _write(overlay, "overlay-manifest.yaml", yaml.safe_dump(manifest))

    rewritten, changed = update.recompute_manifest(overlay, version="new")

    assert changed is True
    assert rewritten["overlay_version"] == "new"
    assert rewritten["components"]["benchmarks/demo"]["tree_sha256"] == audit.tree_sha256(node)
    update.write_manifest(overlay, rewritten)
    assert update.recompute_manifest(overlay)[1] is False


def test_update_overlay_manifest_rejects_unmanifested_files(tmp_path):
    update = _load_manifest_update_module()
    overlay = tmp_path / "overlay"
    node = overlay / "benchmarks/demo"
    _write(node, "config.yaml", "name: demo\n")
    _write(overlay, "benchmarks/stray/extra.py", "print('no')\n")
    manifest = {
        "schema_version": 2,
        "overlay_version": "test",
        "components": {"benchmarks/demo": {"tree_sha256": "stale"}},
        "assets": {},
        "tools": {},
    }
    _write(overlay, "overlay-manifest.yaml", yaml.safe_dump(manifest))

    with pytest.raises(update.ManifestUpdateError, match="outside declared nodes"):
        update.recompute_manifest(overlay)
