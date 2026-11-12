# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asset package loading for synth_env (design §3.5, C-EA-1 consumer side).

A package is an immutable zip produced by DataDesigner (M1: hand-placed). Loading
validates, in order:

1. **fingerprint self-consistency** — the manifest-declared fingerprint equals a
   content-addressed hash recomputed from the zip's own file contents, so any
   byte drift anywhere in the package (replay data, tasks, env_spec, FS
   snapshot) breaks the pin;
2. **triple pin** (ref_id / version / fingerprint) — when the caller supplies an
   expected ``EnvReleasePin`` (from a task row or ``seed_session``), every field
   must match, else the package is rejected (评估前重验, DataDesigner runtime
   pins 语义);
3. **profile gate** — M1 only knows the ``terminal`` profile; unknown profiles
   are rejected loudly instead of mis-executed.

After loading, a ``LoadedPackage`` is treated as read-only for the whole process
lifetime (资产不可变): sessions get deep copies of the FS snapshot and never
write back. ``PackageStore`` holds ``{ref_id: {version: package}}`` so multiple
versions of the same ref coexist and sessions bind to one version at seed time
(design §3.13 决议 9).

The fingerprint is the first 16 hex chars of
``sha256(canonical_json({ref_id, version, profile, files: {path: sha256(file)}}))``
with ``manifest.json`` excluded (it carries the fingerprint itself). If the
DataDesigner publisher later freezes a different definition, only
:func:`compute_fingerprint` changes.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from resources_servers.synth_env.dynamics.replay import ReplayData
from resources_servers.synth_env.task_data import EnvReleasePin, TaskRow


class SynthEnvPackageError(ValueError):
    """A package failed integrity/pin validation and was rejected."""


SUPPORTED_PROFILES = {"terminal", "office"}
MANIFEST_NAME = "manifest.json"

# Office-profile asset files (design §3.5: 状态 schema/规则/NPC 卡/事件模型/
# grounding 索引/llm_sim 模板 are all package-declared). All optional: an office
# package without them degenerates to an empty deterministic skeleton.
OFFICE_JSONL_FILES = {
    "state/seed.jsonl": "state_seed_rows",
    "grounding/index.jsonl": "grounding_rows",
}
OFFICE_YAML_FILES = {
    "rules.yaml": "rules",
    "events.yaml": "event_model",
    "npcs.yaml": "npc_model",
    "llm_sim.yaml": "llm_sim_config",
    "perturbations.yaml": "perturbations",
}

# Perturbation vocabulary mapped from DataDesigner (design §3.9): state_patch
# operations mirror benchmarks/augment.py {op, path, value} shape; failure_probe
# entries are targeted must-reproduce faults. Runtime injection never removes
# objects — variant generation on the asset side owns removal.
PERTURBATION_STATE_PATCH_OPS = {"add", "replace"}


def _validate_perturbations(perturbations: Any) -> None:
    if not isinstance(perturbations, dict):
        raise SynthEnvPackageError(f"perturbations.yaml must be a YAML mapping, got {type(perturbations).__name__}")
    unknown = sorted(set(perturbations) - {"state_patch", "failure_probe"})
    if unknown:
        raise SynthEnvPackageError(f"perturbations.yaml declares unknown sections {unknown}")
    patches = perturbations.get("state_patch", [])
    if not isinstance(patches, list):
        raise SynthEnvPackageError("perturbations.state_patch must be a list of operations")
    for index, operation in enumerate(patches):
        if not isinstance(operation, dict):
            raise SynthEnvPackageError(f"perturbations.state_patch[{index}] must be a mapping")
        if operation.get("op") not in PERTURBATION_STATE_PATCH_OPS:
            raise SynthEnvPackageError(
                f"perturbations.state_patch[{index}].op must be one of {sorted(PERTURBATION_STATE_PATCH_OPS)} "
                f"(remove is owned by asset-side variant generation, not runtime injection)"
            )
        path = operation.get("path")
        if not isinstance(path, list) or not path or not all(isinstance(part, str) for part in path):
            raise SynthEnvPackageError(f"perturbations.state_patch[{index}].path must be a non-empty list of strings")
        if len(path) not in (2, 3):
            raise SynthEnvPackageError(
                f"perturbations.state_patch[{index}].path must be [collection, id] or [collection, id, field]"
            )
        if "value" not in operation:
            raise SynthEnvPackageError(f"perturbations.state_patch[{index}] is missing 'value'")
    probes = perturbations.get("failure_probe", [])
    if not isinstance(probes, list):
        raise SynthEnvPackageError("perturbations.failure_probe must be a list of probe specs")
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            raise SynthEnvPackageError(f"perturbations.failure_probe[{index}] must be a mapping")
        at_step = probe.get("at_step")
        if not isinstance(at_step, int) or isinstance(at_step, bool) or at_step < 1:
            raise SynthEnvPackageError(f"perturbations.failure_probe[{index}].at_step must be a positive integer")
        if not isinstance(probe.get("action"), str) or not probe["action"]:
            raise SynthEnvPackageError(f"perturbations.failure_probe[{index}].action must be a non-empty string")
        if not isinstance(probe.get("message"), str) or not probe["message"]:
            raise SynthEnvPackageError(f"perturbations.failure_probe[{index}].message must be a non-empty string")


class PackageManifest(BaseModel):
    # Frozen: the manifest is the package's identity and is never mutated in place.
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref_id: str
    version: str
    fingerprint: str
    profile: str
    description: str = ""
    # Office packages may pin their simulator model ref (决议 3: 模拟器模型 ref 在
    # 资产包 manifest,与策略模型严格分离); a ModelServerRef-shaped dict.
    sim_model: Optional[dict[str, str]] = None

    @field_validator("version", mode="before")
    @classmethod
    def _normalize_version(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


class EnvSpec(BaseModel):
    """The action contract a package declares (design §3.3/§3.2)."""

    model_config = ConfigDict(extra="forbid")

    action_schema: dict[str, Any]
    command_max_length: int = 2000
    danger_patterns: list[str] = Field(default_factory=list)
    state_schema: dict[str, Any] = Field(default_factory=dict)
    # Office profile: the declared structured actions (§3.11); terminal packages
    # leave it unset and keep the single legacy ``action_schema``.
    actions: Optional[list[dict[str, Any]]] = None
    # Dynamics selection (§3.4): "replay" (M1 default) | "hybrid" (M3 类真实 —
    # allowlisted commands really execute in a controlled subprocess against the
    # materialized seed FS; everything else falls back to replay / is rejected).
    dynamics: str = "replay"
    real_exec_allowlist: list[str] = Field(default_factory=list)
    exec_timeout_s: float = 5.0
    exec_max_output_chars: int = 65536
    hybrid_miss_policy: str = "replay"  # "replay" | "reject"


@dataclass(frozen=True)
class OfficeAssets:
    """The office-profile skeleton assets of one package (all deterministic)."""

    state_seed_rows: list[dict[str, Any]] = field(default_factory=list)
    rules: dict[str, Any] = field(default_factory=dict)
    event_model: dict[str, Any] = field(default_factory=dict)
    npc_model: dict[str, Any] = field(default_factory=dict)
    grounding_rows: list[dict[str, Any]] = field(default_factory=list)
    llm_sim_config: dict[str, Any] = field(default_factory=dict)
    perturbations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedPackage:
    """An immutable loaded package. Sessions reference it; nothing writes to it."""

    manifest: PackageManifest
    env_spec: EnvSpec
    replay: ReplayData
    tasks: dict[str, TaskRow]
    fs_snapshot: dict[str, str]  # absolute path -> text content
    office: Optional[OfficeAssets] = None

    @property
    def pin(self) -> EnvReleasePin:
        return EnvReleasePin(
            ref_id=self.manifest.ref_id, version=self.manifest.version, fingerprint=self.manifest.fingerprint
        )


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_fingerprint(ref_id: str, version: str, profile: str, files: dict[str, str]) -> str:
    """Content-addressed package identity: sha256 over the per-file content hashes.

    ``files`` maps relative path -> file text content. Callers must exclude
    ``manifest.json`` (carries the fingerprint) and ``tasks.jsonl`` (task rows
    embed the fingerprint, so covering them would be circular); task integrity
    is enforced structurally instead (every row must pin the package's own
    ref_id/version/fingerprint).
    """
    payload = {
        "ref_id": ref_id,
        "version": version,
        "profile": profile,
        "files": {path: hashlib.sha256(content.encode("utf-8")).hexdigest() for path, content in sorted(files.items())},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _read_jsonl(raw: str, source: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SynthEnvPackageError(f"{source} line {line_number}: invalid JSON ({error})") from error
    return rows


def load_package_zip(zip_bytes: bytes, *, expect: Optional[EnvReleasePin] = None) -> LoadedPackage:
    """Load and validate a package zip; raises :class:`SynthEnvPackageError` on any failure."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            manifest_raw = archive.read(MANIFEST_NAME).decode("utf-8")
            file_contents = {
                name: archive.read(name).decode("utf-8")
                for name in archive.namelist()
                if name != MANIFEST_NAME and not name.endswith("/")
            }
            env_spec_raw = file_contents["env_spec.json"]
            task_rows = _read_jsonl(file_contents["tasks.jsonl"], "tasks.jsonl")
            snapshot_rows = _read_jsonl(file_contents["fs/snapshot.jsonl"], "fs/snapshot.jsonl")
            fs_snapshot = {row["path"]: row["content"] for row in snapshot_rows}
            # replay is terminal-profile dynamics; office packages ship none.
            replay_rows = (
                _read_jsonl(file_contents["replay/replay.jsonl"], "replay/replay.jsonl")
                if "replay/replay.jsonl" in file_contents
                else []
            )
    except KeyError as error:
        raise SynthEnvPackageError(f"package zip is missing required file {error}") from error
    except zipfile.BadZipFile as error:
        raise SynthEnvPackageError(f"package is not a valid zip: {error}") from error

    try:
        manifest = PackageManifest.model_validate(json.loads(manifest_raw))
    except (json.JSONDecodeError, ValueError) as error:
        raise SynthEnvPackageError(f"invalid {MANIFEST_NAME}: {error}") from error

    if manifest.profile not in SUPPORTED_PROFILES:
        raise SynthEnvPackageError(
            f"unsupported package profile {manifest.profile!r} (this host implements {sorted(SUPPORTED_PROFILES)})"
        )

    recomputed = compute_fingerprint(
        manifest.ref_id,
        manifest.version,
        manifest.profile,
        {name: content for name, content in file_contents.items() if name != "tasks.jsonl"},
    )
    if recomputed != manifest.fingerprint:
        raise SynthEnvPackageError(
            f"fingerprint mismatch for {manifest.ref_id}@v{manifest.version}: "
            f"manifest declares {manifest.fingerprint!r}, content hashes to {recomputed!r}"
        )

    if expect is not None:
        declared = EnvReleasePin(
            ref_id=manifest.ref_id, version=manifest.version, fingerprint=manifest.fingerprint
        )
        if not declared.matches(expect):
            raise SynthEnvPackageError(
                f"pin mismatch: expected {expect.ref_id}@v{expect.version}#{expect.fingerprint}, "
                f"package is {declared.ref_id}@v{declared.version}#{declared.fingerprint}"
            )

    try:
        env_spec = EnvSpec.model_validate(json.loads(env_spec_raw))
    except (json.JSONDecodeError, ValueError) as error:
        raise SynthEnvPackageError(f"invalid env_spec.json: {error}") from error

    office: Optional[OfficeAssets] = None
    if manifest.profile == "office":
        if not env_spec.state_schema:
            raise SynthEnvPackageError("office packages must declare a non-empty env_spec.state_schema")
        if not env_spec.actions:
            raise SynthEnvPackageError("office packages must declare env_spec.actions")
        office = _load_office_assets(file_contents)

    tasks: dict[str, TaskRow] = {}
    for index, row in enumerate(task_rows):
        try:
            task = TaskRow.model_validate(row)
        except ValueError as error:
            raise SynthEnvPackageError(f"tasks.jsonl row {index}: {error}") from error
        if task.env_release.ref_id != manifest.ref_id or task.env_release.version != manifest.version:
            raise SynthEnvPackageError(
                f"task {task.task_id} pins {task.env_release.ref_id}@v{task.env_release.version} "
                f"but the package is {manifest.ref_id}@v{manifest.version}"
            )
        if task.env_release.fingerprint != manifest.fingerprint:
            raise SynthEnvPackageError(f"task {task.task_id} pins fingerprint {task.env_release.fingerprint!r} "
                                       f"but the package fingerprint is {manifest.fingerprint!r}")
        tasks[task.task_id] = task

    return LoadedPackage(
        manifest=manifest,
        env_spec=env_spec,
        replay=ReplayData(replay_rows),
        tasks=tasks,
        fs_snapshot=fs_snapshot,
        office=office,
    )


def _load_office_assets(file_contents: dict[str, str]) -> OfficeAssets:
    """Parse the optional office skeleton files; malformed content is a loud rejection."""
    values: dict[str, Any] = {}
    for name, attr in OFFICE_JSONL_FILES.items():
        if name in file_contents:
            values[attr] = _read_jsonl(file_contents[name], name)
    for name, attr in OFFICE_YAML_FILES.items():
        if name in file_contents:
            try:
                parsed = yaml.safe_load(file_contents[name])
            except yaml.YAMLError as error:
                raise SynthEnvPackageError(f"invalid {name}: {error}") from error
            if parsed is None:
                parsed = {}
            if not isinstance(parsed, dict):
                raise SynthEnvPackageError(f"{name} must be a YAML mapping, got {type(parsed).__name__}")
            values[attr] = parsed
    perturbations = values.get("perturbations", {})
    if perturbations:
        _validate_perturbations(perturbations)
    return OfficeAssets(
        state_seed_rows=values.get("state_seed_rows", []),
        rules=values.get("rules", {}),
        event_model=values.get("event_model", {}),
        npc_model=values.get("npc_model", {}),
        grounding_rows=values.get("grounding_rows", []),
        llm_sim_config=values.get("llm_sim_config", {}),
        perturbations=perturbations,
    )


class PackageBusyError(RuntimeError):
    """Unload was requested while live sessions still reference the version."""


class PackageNotFoundError(KeyError):
    pass


class PackageStore:
    """``{ref_id: {version: LoadedPackage}}`` — same-ref multi-version coexistence (决议 9)."""

    def __init__(self) -> None:
        self._packages: dict[str, dict[str, LoadedPackage]] = {}
        # Wall-clock load time per (ref_id, version) — §3.13 包清单 loaded_at.
        self._loaded_at: dict[tuple[str, str], str] = {}

    def load(self, package: LoadedPackage) -> None:
        versions = self._packages.setdefault(package.manifest.ref_id, {})
        if package.manifest.version in versions:
            raise SynthEnvPackageError(
                f"package {package.manifest.ref_id}@v{package.manifest.version} is already loaded; "
                "loaded packages are immutable and cannot be replaced"
            )
        versions[package.manifest.version] = package
        self._loaded_at[(package.manifest.ref_id, package.manifest.version)] = datetime.now(timezone.utc).isoformat()

    def get(self, pin: EnvReleasePin) -> LoadedPackage:
        package = self._packages.get(pin.ref_id, {}).get(pin.version)
        if package is None:
            raise PackageNotFoundError(
                f"no loaded package {pin.ref_id}@v{pin.version} "
                f"(loaded: {self.list() or 'none'}); load it via the admin API first"
            )
        return package

    def packages(self) -> list[LoadedPackage]:
        """All loaded packages (route registration / admin detail use)."""
        return [
            package
            for ref_id in sorted(self._packages)
            for package in self._packages[ref_id].values()
        ]

    def unload(self, ref_id: str, version: str, active_pins: list[tuple[str, str]]) -> None:
        """Remove a version unless a live session is still bound to it (拒绝，不强杀)."""
        active_count = sum(1 for pin in active_pins if pin == (ref_id, version))
        if active_count:
            raise PackageBusyError(
                f"cannot unload {ref_id}@v{version}: {active_count} "
                "active session(s) reference it; terminate them first"
            )
        versions = self._packages.get(ref_id)
        if not versions or version not in versions:
            raise PackageNotFoundError(f"no loaded package {ref_id}@v{version}")
        del versions[version]
        self._loaded_at.pop((ref_id, version), None)
        if not versions:
            del self._packages[ref_id]

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "ref_id": package.manifest.ref_id,
                "version": package.manifest.version,
                "fingerprint": package.manifest.fingerprint,
                "profile": package.manifest.profile,
                "task_count": len(package.tasks),
                "loaded_at": self._loaded_at.get((package.manifest.ref_id, package.manifest.version)),
            }
            for ref_id in sorted(self._packages)
            for version, package in sorted(self._packages[ref_id].items())
        ]
