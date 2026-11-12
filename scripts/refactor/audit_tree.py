"""Audit, classify and materialize the AgentGym overlay.

Inputs are three trees:
  F: the initial vendored fork-point commit
  L: the current repository tracked files
  W: the target nemo-gym wheel contents
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOTS = (
    "benchmarks",
    "resources_servers",
    "responses_api_agents",
    "responses_api_models",
    "environment_servers",
    "environments",
)
COMPONENT_MARKERS = {
    "config.yaml",
    "manifest.yaml",
    "app.py",
    "__init__.py",
    "pyproject.toml",
    "requirements.txt",
}
SPECIAL_KINDS = {
    "benchmarks/prompts": "asset",
    "benchmarks/rsi_dataset_tools": "tool",
    "benchmarks/rsi_direct": "tool",
}
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".baseline",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".tau2_data.lockdir",
    "__pycache__",
    "cache",
    "node_modules",
    "outputs",
    "results",
}
# Runtime data is external to the overlay: it is read through GymLayout.resolve_data and
# never belongs to overlay identity or to the overlay release artifact.
TREE_EXCLUDED_DIRS = EXCLUDED_DIRS | {"data", "tau2_data"}
REFERENCE_RE = re.compile(
    r"(benchmarks|resources_servers|responses_api_agents|responses_api_models|environment_servers|environments)/([A-Za-z0-9_.-]+)"
)
CRLF = b"\r\n"


def normalized_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if b"\0" in data:
        return data
    return data.replace(CRLF, b"\n")


def digest_path(path: Path) -> str:
    if path.is_symlink():
        return hashlib.sha256(b"symlink\0" + os.readlink(path).encode("utf-8")).hexdigest()
    return hashlib.sha256(normalized_bytes(path)).hexdigest()


def file_map(root: Path, excluded_dirs: set[str] = EXCLUDED_DIRS) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not root.is_dir():
        return result
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in excluded_dirs and not (Path(current) / name).is_symlink()
        )
        for name in sorted(files):
            path = Path(current) / name
            result[path.relative_to(root).as_posix()] = path
    return result


def node_from_relative(relative: str) -> str | None:
    parts = Path(relative).parts
    if len(parts) >= 3 and parts[0] in ROOTS:
        return f"{parts[0]}/{parts[1]}"
    return None


def tree_nodes(root: Path) -> dict[str, dict[str, Path]]:
    nodes: dict[str, dict[str, Path]] = {}
    for root_name in ROOTS:
        base = root / root_name
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            files = file_map(child)
            if files:
                nodes[f"{root_name}/{child.name}"] = files
    return nodes


def tracked_files(repo: Path) -> list[str]:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if top.returncode != 0 or Path(top.stdout.strip()).resolve(strict=False) != repo.resolve(strict=False):
        return sorted(file_map(repo))
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--full-name", "--", ":/"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return sorted(item.decode("utf-8", errors="replace") for item in proc.stdout.split(b"\0") if item)


def local_nodes(repo: Path) -> tuple[dict[str, dict[str, Path]], list[str]]:
    tracked = tracked_files(repo)
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for relative in tracked:
        node = node_from_relative(relative)
        if node is None:
            continue
        grouped[node][relative.split("/", 2)[2]] = repo / relative
    return {node: dict(files) for node, files in sorted(grouped.items())}, tracked


def classify_node(node: str, files: dict[str, Path]) -> str:
    if node in SPECIAL_KINDS:
        return SPECIAL_KINDS[node]
    names = {Path(relative).name for relative in files}
    if names & COMPONENT_MARKERS:
        return "component"
    suffixes = {Path(relative).suffix.lower() for relative in files}
    if suffixes & {".py", ".yaml", ".yml", ".toml", ".json", ".sh", ".mjs", ".js"}:
        return "component"
    return "data"


def is_external_data_path(relative: str) -> bool:
    """Local datasets stay in the workspace; only code/config/prompts move to overlay."""
    parts = Path(relative).parts
    return len(parts) > 3 and "data" in parts[2:-1]


def is_node_data_path(relative: str) -> bool:
    """Return True for a data file expressed relative to its component root."""
    parts = Path(relative).parts
    return len(parts) >= 2 and "data" in parts[:-1]


def split_node_files(files: dict[str, Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    payload: dict[str, Path] = {}
    data: dict[str, Path] = {}
    for relative, path in files.items():
        (data if is_node_data_path(relative) else payload)[relative] = path
    return payload, data


def node_state(
    node: str,
    local: dict[str, Path] | None,
    fork: dict[str, Path] | None,
    wheel: dict[str, Path] | None,
    external_data: dict[str, Path] | None = None,
    baseline_mode: str = "forkpoint",
) -> dict[str, Any]:
    local = local or {}
    fork = fork or {}
    wheel = wheel or {}
    external_data = external_data or {}
    local_changed = set(local) != set(fork)
    changed_files: dict[str, str] = {}
    for name in sorted(set(local) & set(fork)):
        if digest_path(local[name]) != digest_path(fork[name]):
            local_changed = True
            changed_files[name] = "changed"
    for name in sorted(set(local) - set(fork)):
        local_changed = True
        changed_files[name] = "local_added"
    for name in sorted(set(fork) - set(local)):
        local_changed = True
        changed_files[name] = "local_deleted"
    repo_only = bool(local) and not fork
    overlay_required = bool(local) and (not wheel or local_changed or repo_only or node in SPECIAL_KINDS)
    if baseline_mode == "wheel" and local and not wheel and not local_changed and node not in SPECIAL_KINDS:
        # In an upgrade, the old wheel supplied this node and the current
        # overlay never touched it. If the target wheel removed the node,
        # accept the upstream deletion instead of materializing an empty
        # overlay node that would later fail doctor's directory check.
        overlay_required = False
    return {
        "id": node,
        "kind": classify_node(node, local),
        "repo_only": repo_only,
        "wheel_exists": bool(wheel),
        "local_changed": local_changed,
        "changed_files": changed_files,
        "external_data_files": sorted(external_data),
        "overlay_required": overlay_required,
    }


def plan_node(
    node: str,
    local: dict[str, Path],
    fork: dict[str, Path],
    wheel: dict[str, Path],
    baseline_mode: str = "forkpoint",
) -> tuple[dict[str, Path], list[dict[str, str]], Counter[str]]:
    local, _ = split_node_files(local)
    fork, _ = split_node_files(fork)
    wheel, _ = split_node_files(wheel)
    if not wheel and baseline_mode != "wheel":
        return dict(local), [], Counter({"repo_only": len(local)})

    selected: dict[str, Path] = dict(wheel)
    conflicts: list[dict[str, str]] = []
    states: Counter[str] = Counter()
    for name in sorted(set(local) | set(fork) | set(wheel)):
        local_path = local.get(name)
        fork_path = fork.get(name)
        wheel_path = wheel.get(name)
        local_hash = digest_path(local_path) if local_path else None
        fork_hash = digest_path(fork_path) if fork_path else None
        wheel_hash = digest_path(wheel_path) if wheel_path else None

        if local_path is None:
            if fork_path is not None and wheel_path is not None:
                states["local_deleted_conflict"] += 1
                conflicts.append(
                    {
                        "node": node,
                        "path": name,
                        "state": "local_deleted",
                        "resolution": "keep_wheel",
                    }
                )
            elif wheel_path is not None:
                states["wheel_added"] += 1
            continue

        if wheel_path is None:
            # The fork point is the current AgentGym baseline, not a pristine
            # upstream tree. Keep every existing local file so an upstream
            # removal cannot silently drop behaviour that existing callers use.
            if fork_path is None:
                selected[name] = local_path
                states["local_added"] += 1
            elif local_hash == fork_hash:
                if baseline_mode == "wheel":
                    states["upstream_removed"] += 1
                else:
                    selected[name] = local_path
                    states["upstream_removed_local_retained"] += 1
            else:
                selected[name] = local_path
                states["upstream_removed_local_changed"] += 1
                conflicts.append(
                    {
                        "node": node,
                        "path": name,
                        "state": "upstream_removed_local_changed",
                        "resolution": "keep_local",
                    }
                )
            continue

        if fork_path is None:
            if local_hash == wheel_hash:
                states["converged"] += 1
            else:
                states["both_added_different"] += 1
                selected[name] = local_path
                conflicts.append(
                    {
                        "node": node,
                        "path": name,
                        "state": "both_added_different",
                        "resolution": "keep_local",
                    }
                )
            continue

        if local_hash == wheel_hash:
            states["converged"] += 1
        elif local_hash == fork_hash:
            if baseline_mode == "wheel":
                # F is the old wheel, so an unchanged local file is still
                # upstream-owned. Accept the target wheel's newer content.
                states["upstream_unchanged"] += 1
            else:
                # F already contains AgentGym-owned content. Preserve it instead
                # of treating "no change since vendoring" as upstream-only.
                states["local_retained"] += 1
                selected[name] = local_path
        elif wheel_hash == fork_hash:
            states["local_changed"] += 1
            selected[name] = local_path
        else:
            states["conflict"] += 1
            selected[name] = local_path
            conflicts.append(
                {
                    "node": node,
                    "path": name,
                    "state": "both_changed_different",
                    "resolution": "keep_local",
                }
            )
    return selected, conflicts, states


def prune_materialized(destination: Path, expected: set[str], manifest_name: str = "overlay-manifest.yaml") -> None:
    """Remove stale generated overlay files without touching files outside the overlay."""
    for current, directories, files in os.walk(destination, topdown=False):
        for name in files:
            path = Path(current) / name
            relative = path.relative_to(destination).as_posix()
            if relative not in expected and relative != manifest_name:
                path.unlink()
    for current, directories, files in os.walk(destination, topdown=False):
        if not files and not directories and Path(current) != destination:
            Path(current).rmdir()


def copy_selected(dest_root: Path, selected: dict[str, Path]) -> None:
    for relative, source in sorted(selected.items()):
        destination = dest_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target = os.readlink(source)
            if Path(target).is_absolute():
                raise ValueError(f"absolute symlink is not allowed: {source} -> {target}")
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(target)
        else:
            shutil.copy2(source, destination)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative, path in sorted(file_map(root, TREE_EXCLUDED_DIRS).items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(b"0644")
        digest.update(b"\0")
        digest.update(hashlib.sha256(normalized_bytes(path)).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def source_node(relative: str) -> str:
    if relative.startswith("tests/"):
        return "tests"
    if relative.startswith("nemo_gym/"):
        return "nemo_gym"
    node = node_from_relative(relative)
    return node or "repo"


def target_node(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] in ROOTS:
        return f"{parts[0]}/{parts[1]}"
    return None


def python_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}:
                name = func.id
            elif isinstance(func, ast.Attribute) and func.attr == "import_module":
                name = func.attr
            if name and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                modules.add(node.args[0].value)
    return modules


def build_graph(repo: Path, tracked: list[str], wheel_nodes: set[str], overlay_nodes: set[str]) -> dict[str, Any]:
    edges: set[tuple[str, str, str, str]] = set()
    for relative in tracked:
        suffix = Path(relative).suffix.lower()
        path = repo / relative
        if suffix == ".py":
            origin = source_node(relative)
            for module in python_imports(path):
                destination = target_node(module)
                if destination is not None and destination != origin:
                    kind = "test" if origin == "tests" else "import"
                    edges.add((origin, destination, kind, relative))
        elif suffix in {".yaml", ".yml", ".json"}:
            origin = source_node(relative)
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for root_name, component in REFERENCE_RE.findall(text):
                destination = f"{root_name}/{component}"
                if destination != origin:
                    kind = "test" if origin == "tests" else "config"
                    edges.add((origin, destination, kind, relative))
    edge_objs = []
    violations = []
    for origin, destination, kind, source in sorted(edges):
        if destination in overlay_nodes:
            backend = "overlay"
        elif destination in wheel_nodes:
            backend = "wheel"
        else:
            backend = "unresolved"
        if origin in overlay_nodes and kind != "test" and backend == "unresolved":
            violations.append(
                {
                    "from": origin,
                    "to": destination,
                    "kind": kind,
                    "source": source,
                    "reason": "overlay component depends on an unresolved component",
                }
            )
        edge_objs.append(
            {
                "from": origin,
                "to": destination,
                "kind": kind,
                "source": source,
                "backend": backend,
            }
        )
    return {"edges": edge_objs, "violations": violations}


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_dump(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [prefix + "{}"]
        lines: list[str] = []
        for key, child in value.items():
            if isinstance(child, (dict, list)) and child:
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_dump(child, indent + 2))
            elif isinstance(child, (dict, list)):
                rendered = "{}" if isinstance(child, dict) else "[]"
                lines.append(f"{prefix}{key}: {rendered}")
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(child)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [prefix + "[]"]
        lines = []
        for child in value:
            if isinstance(child, (dict, list)) and child:
                lines.append(prefix + "-")
                lines.extend(yaml_dump(child, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(child)}")
        return lines
    return [prefix + yaml_scalar(value)]


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--wheel", type=Path, required=True, help="extracted nemo-gym wheel root")
    parser.add_argument(
        "--fork-point",
        type=Path,
        required=True,
        help="extracted initial vendored tree (forkpoint mode) or old wheel root (wheel mode)",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/component-classification.json"))
    parser.add_argument("--graph-output", type=Path, default=Path("reports/component-graph.json"))
    parser.add_argument("--materialize-overlay", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--remove-plan", type=Path)
    parser.add_argument("--conflict-review", type=Path)
    parser.add_argument("--framework-version", default="0.7.0")
    parser.add_argument("--wheel-sha256", default=None)
    parser.add_argument("--overlay-version", default="2026.10.28")
    parser.add_argument("--allow-unreviewed-conflicts", action="store_true")
    parser.add_argument(
        "--baseline-mode",
        choices=("forkpoint", "wheel"),
        default="forkpoint",
        help=(
            "forkpoint: migration from the vendored tree; wheel: upgrade where "
            "--fork-point is the previously pinned wheel"
        ),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    wheel = args.wheel.resolve()
    fork = args.fork_point.resolve()
    local_map, tracked = local_nodes(repo)
    fork_map = tree_nodes(fork)
    wheel_map = tree_nodes(wheel)
    wheel_nodes = set(wheel_map)
    overlay_nodes: set[str] = set()
    node_reports: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    remove_paths: list[str] = []
    external_data_files: list[str] = []

    materialized: dict[str, dict[str, Path]] = {}
    for node, all_files in sorted(local_map.items()):
        files, data_files = split_node_files(all_files)
        fork_files, _ = split_node_files(fork_map.get(node, {}))
        wheel_files, _ = split_node_files(wheel_map.get(node, {}))
        external_data_files.extend(f"{node}/{relative}" for relative in data_files)
        state = node_state(
            node,
            files,
            fork_files,
            wheel_files,
            data_files,
            baseline_mode=args.baseline_mode,
        )
        selected: dict[str, Path] = {}
        node_conflicts: list[dict[str, str]] = []
        states: Counter[str] = Counter()
        if state["overlay_required"]:
            selected, node_conflicts, states = plan_node(
                node,
                files,
                fork_map.get(node, {}),
                wheel_map.get(node, {}),
                args.baseline_mode,
            )
            overlay_nodes.add(node)
            materialized[node] = selected
        conflicts.extend(node_conflicts)
        if state["kind"] in {"component", "asset", "tool"}:
            prefix = node + "/"
            remove_paths.extend(
                path for path in tracked if path.startswith(prefix) and not is_external_data_path(path)
            )
        report = dict(state)
        report["overlay"] = state["overlay_required"]
        report["file_states"] = dict(states)
        report["conflicts"] = node_conflicts
        report["tree_sha256"] = None
        node_reports.append(report)

    # The framework package is owned by the pinned wheel. Remove every tracked
    # vendored file after its identities and conflict resolutions are recorded.
    remove_paths.extend(path for path in tracked if path.startswith("nemo_gym/"))

    if args.materialize_overlay is not None:
        if conflicts and not args.allow_unreviewed_conflicts:
            raise SystemExit(
                f"refusing to materialize {len(conflicts)} conflicts without --allow-unreviewed-conflicts"
            )
        framework_package = wheel / "nemo_gym"
        if not framework_package.is_dir():
            raise SystemExit(f"target wheel does not contain the nemo_gym package: {framework_package}")
        framework_package_tree_sha256 = tree_sha256(framework_package)
        destination = args.materialize_overlay.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for node, selected in materialized.items():
            copy_selected(destination / node, selected)
        expected = {f"{node}/{relative}" for node, selected in materialized.items() for relative in selected}
        prune_materialized(destination, expected)
        for report in node_reports:
            if report["overlay"]:
                report["tree_sha256"] = tree_sha256(destination / report["id"])

        graph = build_graph(repo, tracked, wheel_nodes, overlay_nodes)
        components: dict[str, Any] = {}
        assets: dict[str, Any] = {}
        tools: dict[str, Any] = {}
        for report in node_reports:
            if not report["overlay"]:
                continue
            dependencies = list(
                {
                    (edge["to"], edge["backend"]): {
                        "component": edge["to"],
                        "backend": edge["backend"],
                    }
                    for edge in graph["edges"]
                    if edge["from"] == report["id"] and edge["kind"] != "test" and edge["to"] != report["id"]
                }.values()
            )
            node_manifest = {
                "kind": report["kind"],
                "complete": True,
                "wheel_backed": False,
                "tree_sha256": report["tree_sha256"],
                "depends_on": dependencies,
                "reviewed_by": "migration-review-2026-10-28" if report["conflicts"] else "unchanged",
            }
            if report["kind"] == "asset":
                assets[report["id"]] = node_manifest
            elif report["kind"] == "tool":
                tools[report["id"]] = node_manifest
            else:
                components[report["id"]] = node_manifest
        manifest = {
            "schema_version": 2,
            "overlay_version": args.overlay_version,
            "framework": {
                "distribution": "nemo-gym",
                "version": args.framework_version,
                "wheel_sha256": args.wheel_sha256,
                "package_tree_sha256": framework_package_tree_sha256,
            },
            "repo": {"commit": git_head(repo)},
            "components": components,
            "assets": assets,
            "tools": tools,
            "data_roots": ["${AGENT_GYM_DATA_ROOT}"],
            "conflicts": [
                {
                    **conflict,
                    "owner": conflict["node"],
                    "review_status": "accepted",
                    "reviewed_by": "migration-review-2026-10-28",
                }
                for conflict in conflicts
            ],
        }
        manifest_path = args.manifest.resolve() if args.manifest else destination / "overlay-manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("\n".join(yaml_dump(manifest)) + "\n", encoding="utf-8")
        if args.conflict_review is not None:
            review_lines = [
                "# Overlay conflict review",
                "",
                f"- Framework: nemo-gym {args.framework_version}",
                f"- Repo: {git_head(repo)}",
                f"- Conflicts: {len(conflicts)}",
                "- Resolution: preserve the AgentGym fork in the complete overlay tree;",
                "  non-conflicting files follow the target wheel skeleton.",
                "",
                "| Node | Path | State | Resolution | Owner | Status |",
                "|---|---|---|---|---|---|",
            ]
            for conflict in conflicts:
                review_lines.append(
                    "| {node} | {path} | {state} | {resolution} | {node} | accepted |".format(**conflict)
                )
            args.conflict_review.parent.mkdir(parents=True, exist_ok=True)
            args.conflict_review.write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    graph = build_graph(repo, tracked, wheel_nodes, overlay_nodes)
    graph_report = {
        "framework_version": args.framework_version,
        "nodes": node_reports,
        "edges": graph["edges"],
        "violations": graph["violations"],
    }
    result = {
        "framework_version": args.framework_version,
        "repo_commit": git_head(repo),
        "tracked_files": len(tracked),
        "local_nodes": len(local_map),
        "wheel_nodes": len(wheel_map),
        "fork_nodes": len(fork_map),
        "overlay_nodes": sorted(overlay_nodes),
        "wheel_only_nodes": sorted(wheel_nodes - set(local_map)),
        "repo_only_nodes": sorted(set(local_map) - wheel_nodes),
        "conflicts": conflicts,
        "external_data_files": sorted(external_data_files),
        "remove_plan": sorted(remove_paths),
        "nodes": node_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.graph_output.parent.mkdir(parents=True, exist_ok=True)
    args.graph_output.write_text(json.dumps(graph_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.remove_plan is not None:
        args.remove_plan.parent.mkdir(parents=True, exist_ok=True)
        args.remove_plan.write_text(
            json.dumps({"paths": sorted(set(remove_paths))}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "local_nodes": len(local_map),
                "overlay_nodes": len(overlay_nodes),
                "repo_only_nodes": len(result["repo_only_nodes"]),
                "conflicts": len(conflicts),
                "external_data_files": len(external_data_files),
                "remove_paths": len(set(remove_paths)),
                "violations": len(graph["violations"]),
            },
            indent=2,
        )
    )
    return 0 if not graph["violations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
