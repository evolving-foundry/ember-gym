# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Harness manifest (HF-1/HF-3): content-addressed sha256 over everything a score depends on.

The digest covers the adapter harness config, benchmark config, model server pin, tier preset,
profile id, dataset version, and the Gym commit — so any harness-side drift changes the digest and
breaks cross-lineage comparison instead of silently mixing results.
"""

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from agent_gym.layout import GymLayout
from eval_service.contract import EvalRequest
from eval_service.profile import ExperimentProfile


MANIFEST_PREFIX = "sha256:"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_path(
    fpath: str,
    repo_root: Path,
    layout: Optional[GymLayout],
    *,
    data: bool = False,
) -> Path:
    if layout is not None:
        resolver = layout.resolve_data if data else layout.resolve
        return resolver(fpath, must_exist=False)
    return (repo_root / fpath).resolve()


def _file_hash(
    fpath: str,
    repo_root: Path,
    layout: Optional[GymLayout] = None,
    *,
    data: bool = False,
) -> Optional[str]:
    path = _resolve_path(fpath, repo_root, layout, data=data)
    if not path.is_file():
        return None
    stat = path.stat()
    return _cached_file_hash(str(path), stat.st_mtime_ns, stat.st_size)


def _source_hash(
    source_path: str,
    repo_root: Path,
    layout: Optional[GymLayout] = None,
) -> Optional[str]:
    """Hash one source file or a deterministic source-only tree snapshot."""
    path = _resolve_path(source_path, repo_root, layout)
    if path.is_file():
        return _file_hash(source_path, repo_root, layout)
    if not path.is_dir():
        return None
    included_suffixes = {".py", ".yaml", ".yml", ".toml", ".sh", ".txt", ".js", ".mjs", ".json"}
    excluded_parts = {
        ".git",
        ".venv",
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
    files: list[Path] = []
    for root, directories, names in os.walk(path, topdown=True, followlinks=False):
        # Prune ignored trees before descending. Filtering after Path.rglob still walks
        # generated benchmark data, which can be large and lives on slow Windows mounts.
        directories[:] = sorted(directory for directory in directories if directory not in excluded_parts)
        for name in sorted(names):
            file = Path(root) / name
            if name in {"env.yaml", ".env"}:
                continue
            if file.suffix.lower() in included_suffixes or file.name == "Dockerfile":
                files.append(file)
    files.sort()
    digest = hashlib.sha256()
    for file in files:
        relative = file.relative_to(path).as_posix()
        stat = file.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_cached_file_hash(str(file), stat.st_mtime_ns, stat.st_size)))
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _cached_file_hash(path: str, mtime_ns: int, size: int) -> str:
    # Dataset JSONLs can be large. Cache by resolved path and stat identity so a
    # profile's repeated manifest lookups do not re-hash the entire task pack.
    del mtime_ns, size
    return _hash_bytes(Path(path).read_bytes())


def resolve_gym_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, timeout=10, check=True)
        return out.stdout.decode(errors="replace").strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


def _environment_hashes(names: list[str]) -> dict[str, Optional[str]]:
    """Pin non-secret runtime settings without recording their raw values."""
    return {name: _hash_bytes(os.environ[name].encode("utf-8")) if name in os.environ else None for name in names}


def build_harness_manifest(
    profile: ExperimentProfile,
    request: EvalRequest,
    repo_root: Path,
    gym_commit: Optional[str] = None,
    *,
    layout: Optional[GymLayout] = None,
) -> str:
    """Return the "sha256:..." manifest digest for this (profile, request) pair."""
    adapter = profile.adapters[request.adapter]
    benchmark = profile.benchmarks[request.benchmark]
    tier = profile.tier_preset(request.tier)
    dataset_version = request.dataset_version if request.dataset_version != "auto" else benchmark.dataset_version

    components = {
        "profile_id": profile.profile_id,
        "gym_commit": gym_commit if gym_commit is not None else resolve_gym_commit(repo_root),
        "adapter": {
            "harness_agent": adapter.harness_agent,
            "version": adapter.version,
            "config_hashes": {p: _file_hash(p, repo_root, layout) for p in adapter.config_paths},
            "source_hashes": {p: _source_hash(p, repo_root, layout) for p in adapter.source_paths},
            "environment_hashes": _environment_hashes(adapter.manifest_env),
        },
        "benchmark": {
            "name": request.benchmark,
            "dataset_version": dataset_version,
            "dataset_sha256": _file_hash(benchmark.dataset_fpath, repo_root, layout, data=True),
            "holdout_dataset_sha256": (
                _file_hash(benchmark.holdout_dataset_fpath, repo_root, layout, data=True)
                if benchmark.holdout_dataset_fpath
                else None
            ),
            "harness_agent": adapter.harness_agent,
            "task_id_field": benchmark.task_id_field,
            "config_hashes": {p: _file_hash(p, repo_root, layout) for p in benchmark.config_paths},
            "source_hashes": {p: _source_hash(p, repo_root, layout) for p in benchmark.source_paths},
            "environment_hashes": _environment_hashes(benchmark.manifest_env),
            # §3.3 (HF-1): the frozen rubric library snapshot joins the manifest — any snapshot
            # drift (criteria or review metadata) changes the digest and breaks cross-lineage
            # comparison instead of silently mixing scores from different criteria libraries.
            "rubric": (
                {
                    "library_fpath": benchmark.rubric_library_fpath,
                    "library_file_sha256": _file_hash(benchmark.rubric_library_fpath, repo_root, layout, data=True),
                    "group_size": benchmark.rubric_group_size,
                    "judge": benchmark.rubric_judge.model_dump() if benchmark.rubric_judge else None,
                    "environment_hashes": _environment_hashes(benchmark.rubric_judge.manifest_env)
                    if benchmark.rubric_judge
                    else {},
                }
                if benchmark.rubric_library_fpath is not None
                else None
            ),
        },
        "model": profile.model.model_dump() if profile.model else None,
        "tier": {
            "name": request.tier.value,
            "limit": tier.limit,
            "num_repeats": tier.num_repeats,
            "concurrency": tier.concurrency,
        },
    }
    return MANIFEST_PREFIX + _hash_bytes(_canonical(components))
