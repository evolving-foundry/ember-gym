# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State engine (design §3.2 六件套): schema-checked object store for the office profile.

The office environment's deterministic skeleton: tickets / mailbox / documents /
ledger live as objects whose shape is declared by the package's ``env_spec
.state_schema``. Every mutation — whether it comes from an agent action, a rule,
an event injection or an NPC turn — passes field-level validation before it
touches the store, so a generative layer (llm_sim) can only ever fill *content
fields* and can never alter the structure (状态迁移过 schema 校验).

Field visibility (``visible_to``) powers the role-visibility layer (§3.2 视角/
权限): observations shown to the agent are cropped to the fields its role is
allowed to see — the anti-cheat property that keeps the agent from "reading the
backing store" instead of doing the task.
"""

from __future__ import annotations

from typing import Any, Optional

FIELD_TYPES = {"string", "number", "boolean", "enum"}

# Observation-layer role that sees every field (§3.9 visibility=expose_all).
# Cropping stays the observation layer's duty — the backing store is untouched.
EXPOSE_ALL_ROLE = "*"


class StateSchemaError(ValueError):
    """A mutation that violates the state schema (illegal write) — always observable."""


def _check_value(field_spec: dict[str, Any], value: Any, where: str) -> None:
    ftype = field_spec.get("type", "string")
    if ftype not in FIELD_TYPES:
        raise StateSchemaError(f"{where}: unknown field type {ftype!r}")
    if ftype == "enum":
        allowed = field_spec.get("enum")
        if not isinstance(allowed, list) or not allowed:
            raise StateSchemaError(f"{where}: enum field must declare a non-empty 'enum' list")
        if value not in allowed:
            raise StateSchemaError(f"{where}: value {value!r} is not one of {allowed}")
        return
    if ftype == "string" and not isinstance(value, str):
        raise StateSchemaError(f"{where}: expected string, got {type(value).__name__}")
    if ftype == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StateSchemaError(f"{where}: expected number, got {type(value).__name__}")
    if ftype == "boolean" and not isinstance(value, bool):
        raise StateSchemaError(f"{where}: expected boolean, got {type(value).__name__}")


class StateEngine:
    """Object store ``{kind: {key: object}}`` governed by the package state schema."""

    def __init__(self, schema: dict[str, Any], seed_rows: Optional[list[dict[str, Any]]] = None):
        self._schema = schema or {}
        self._objects: dict[str, dict[str, dict[str, Any]]] = {kind: {} for kind in self._schema}
        self._unknown_kinds: set[str] = set()
        for row in seed_rows or []:
            kind = row.get("kind")
            obj = row.get("obj")
            if kind not in self._schema:
                raise StateSchemaError(f"seed row references unknown collection {kind!r}")
            if not isinstance(obj, dict):
                raise StateSchemaError(f"seed row for {kind!r} has no object")
            self.create(kind, obj)

    # ---- schema access -----------------------------------------------------

    def collection_schema(self, kind: str) -> dict[str, Any]:
        spec = self._schema.get(kind)
        if spec is None:
            raise StateSchemaError(f"unknown collection {kind!r} (schema declares {sorted(self._schema)})")
        return spec

    def key_field(self, kind: str) -> str:
        return self.collection_schema(kind).get("key", "id")

    # ---- mutation (all writes pass validation, no exceptions) ---------------

    def create(self, kind: str, obj: dict[str, Any]) -> dict[str, Any]:
        """Validate a whole object and insert it; duplicate keys are rejected."""
        spec = self.collection_schema(kind)
        fields = spec.get("fields", {})
        key_field = spec.get("key", "id")
        key = obj.get(key_field)
        if not isinstance(key, str) or not key:
            raise StateSchemaError(f"{kind}: object must carry a non-empty {key_field!r}")
        known = set(fields)
        unknown = sorted(set(obj) - known)
        if unknown:
            raise StateSchemaError(f"{kind}[{key}]: fields {unknown} are not in the state schema")
        for name, field_spec in fields.items():
            where = f"{kind}[{key}].{name}"
            if name in obj:
                _check_value(field_spec, obj[name], where)
            elif field_spec.get("required"):
                raise StateSchemaError(f"{where}: required field missing")
        bucket = self._objects[kind]
        if key in bucket:
            raise StateSchemaError(f"{kind}[{key}]: object already exists")
        bucket[key] = dict(obj)
        return dict(obj)

    def patch(self, kind: str, key: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate a field-level update against the schema, then apply it."""
        spec = self.collection_schema(kind)
        key_field = spec.get("key", "id")
        if key_field in patch:
            raise StateSchemaError(f"{kind}: the key field {key_field!r} is immutable")
        obj = self._objects.get(kind, {}).get(key)
        if obj is None:
            raise StateSchemaError(f"{kind}[{key}]: no such object")
        fields = spec.get("fields", {})
        unknown = sorted(set(patch) - set(fields))
        if unknown:
            raise StateSchemaError(f"{kind}[{key}]: fields {unknown} are not in the state schema")
        for name, value in patch.items():
            _check_value(fields[name], value, f"{kind}[{key}].{name}")
        obj.update(patch)
        return dict(obj)

    # ---- reads ---------------------------------------------------------------

    def get(self, kind: str, key: str) -> Optional[dict[str, Any]]:
        obj = self._objects.get(kind, {}).get(key)
        return dict(obj) if obj is not None else None

    def list_kind(self, kind: str) -> list[dict[str, Any]]:
        return [dict(obj) for _, obj in sorted(self._objects.get(kind, {}).items())]

    def find(self, kind: str, match: dict[str, Any]) -> Optional[dict[str, Any]]:
        """First (sorted-key) object whose fields equal all of ``match``."""
        for key in sorted(self._objects.get(kind, {})):
            obj = self._objects[kind][key]
            if all(obj.get(field) == value for field, value in match.items()):
                return dict(obj)
        return None

    def visible_view(self, kind: str, role: str) -> list[dict[str, Any]]:
        """Objects cropped to the fields ``role`` may see (视角/权限, §3.2).

        A field without ``visible_to`` is world-visible; with ``visible_to``, only
        the listed roles observe it. Cropping is the observation-layer duty — the
        backing store always holds the full object.
        """
        fields = self.collection_schema(kind).get("fields", {})
        if role == EXPOSE_ALL_ROLE:
            allowed = set(fields)
        else:
            allowed = {
                name for name, spec in fields.items()
                if not spec.get("visible_to") or role in spec["visible_to"]
            }
        view = []
        for obj in self.list_kind(kind):
            view.append({name: value for name, value in obj.items() if name in allowed})
        return view

    def visible_view_summary(self, role: str) -> dict[str, list[dict[str, Any]]]:
        return {kind: self.visible_view(kind, role) for kind in sorted(self._objects)}

    def summary(self, role: str = "agent") -> str:
        """Deterministic one-line-per-collection digest for state_summary."""
        lines = []
        for kind in sorted(self._objects):
            view = self.visible_view(kind, role)
            lines.append(f"{kind}: {len(view)} object(s)")
        return "\n".join(lines) if lines else "(empty state)"
