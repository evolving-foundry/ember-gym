# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Office profile (M2, design §3.2 六件套): structured actions over a deterministic core.

Principle: **the LLM is caged as a content generator; state, rules and events are
all deterministic.** This profile owns:

- action parsing against the package's declared action list (multi-tool, §3.11
  profile-semantic routes);
- effect execution: fsm transitions / object creation / document writes, every
  mutation passing the state schema (LLM output can fill content fields only);
- role visibility: observations are cropped to what the agent's role may see —
  the first priority of evaluation validity (§3.2 视角/权限);
- the step pipeline: rules adjudication → schema-checked mutation → event tick →
  NPC turns (llm_sim content, rules-adjudicated validity) → observation.

Scenario semantics (tickets, approvals, 两票) live entirely in the asset
package; this module is a generic interpreter over the declared effects.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from resources_servers.synth_env.engine.events import FiredEvent
from resources_servers.synth_env.engine.npc import NpcTurnRequest
from resources_servers.synth_env.engine.state import StateEngine, StateSchemaError
from resources_servers.synth_env.package_loader import EnvSpec
from resources_servers.synth_env.profiles.terminal import SynthEnvActionError, VirtualFS

AGENT_ROLE = "agent"

_ARG_TYPES = {"string": str, "number": (int, float), "boolean": bool}


class OfficeProfile:
    """M2 profile: multi-action office environment over the deterministic skeleton."""

    def __init__(self, env_spec: EnvSpec, root_files: dict[str, str]):
        self.env_spec = env_spec
        self.root_files = root_files
        self._actions: dict[str, dict[str, Any]] = {
            definition["name"]: definition for definition in (env_spec.actions or [])
        }

    # ---- actions ------------------------------------------------------------

    def action_names(self) -> list[str]:
        return sorted(self._actions)

    def action_schemas(self) -> list[dict[str, Any]]:
        """Tool declarations the strategy side mirrors (§3.11 MCP tool schemas)."""
        return [
            {
                "name": name,
                "description": definition.get("description", ""),
                "parameters": definition.get("parameters", {"type": "object", "properties": {}}),
            }
            for name, definition in sorted(self._actions.items())
        ]

    def parse_action(self, action: Any) -> tuple[str, dict[str, Any]]:
        """Validate ``{"name", "args"}`` against the declared action list."""
        if not isinstance(action, dict):
            raise SynthEnvActionError(f"action must be an object, got {type(action).__name__}")
        name = action.get("name")
        definition = self._actions.get(name)
        if definition is None:
            raise SynthEnvActionError(
                f"unknown action {name!r}; declared actions: {self.action_names()}"
            )
        args = action.get("args")
        if not isinstance(args, dict):
            args = {}
        payload = json.dumps(args, ensure_ascii=False)
        if len(payload) > self.env_spec.command_max_length:
            raise SynthEnvActionError(
                f"action args exceed the {self.env_spec.command_max_length}-character limit"
            )
        parameters = definition.get("parameters") or {}
        properties = parameters.get("properties", {})
        unknown = sorted(set(args) - set(properties))
        if unknown:
            raise SynthEnvActionError(f"action {name!r} got undeclared arguments {unknown}")
        for required in parameters.get("required", []):
            if required not in args:
                raise SynthEnvActionError(f"action {name!r} is missing required argument {required!r}")
        for arg_name, value in args.items():
            spec = properties.get(arg_name, {})
            expected = spec.get("type", "string")
            if expected == "enum":
                allowed = spec.get("enum", [])
                if value not in allowed:
                    raise SynthEnvActionError(f"argument {arg_name!r} must be one of {allowed}")
                continue
            python_type = _ARG_TYPES.get(expected)
            if python_type is None:
                raise SynthEnvActionError(f"argument {arg_name!r} declares unknown type {expected!r}")
            if expected == "number" and isinstance(value, bool):
                raise SynthEnvActionError(f"argument {arg_name!r} must be a number")
            if not isinstance(value, python_type):
                raise SynthEnvActionError(f"argument {arg_name!r} must be a {expected}")
        return name, args

    # ---- session scaffolding --------------------------------------------------

    def snapshot(self) -> VirtualFS:
        return VirtualFS(self.root_files)

    def initial_obs(self, goal: str, state: StateEngine, role: str = AGENT_ROLE) -> tuple[dict[str, str], str]:
        collections = "\n".join(
            f"{kind}: " + json.dumps(view, ensure_ascii=False)
            for kind, view in sorted(state.visible_view_summary(role).items())
        )
        obs = {
            "channel": "text",
            "text": (
                f"TASK: {goal}\n\nAvailable actions: {self.action_names()}\n\n"
                f"Workspace state:\n{collections or '(empty)'}"
            ),
        }
        return obs, state.summary(role)

    def rejection_obs(self, message: str) -> dict[str, str]:
        return {"channel": "text", "text": f"synth_env: action rejected — {message}"}

    def build_obs(
        self,
        *,
        result: str,
        npc_lines: list[str],
        event_lines: list[str],
        fs: VirtualFS,
        baseline: VirtualFS,
    ) -> dict[str, str]:
        parts = [result]
        parts.extend(f"[event] {line}" for line in event_lines)
        parts.extend(f"[npc] {line}" for line in npc_lines)
        diff = fs.diff_summary(baseline)
        if diff and diff != "(no file changes)":
            parts.append(f"file changes:\n{diff}")
        return {"channel": "text", "text": "\n".join(part for part in parts if part)}

    # ---- step pipeline ----------------------------------------------------------

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session: Any,
    ) -> dict[str, Any]:
        """Run one agent action through the office pipeline.

        ``session`` is the host's per-session office context (duck-typed):
        ``state`` StateEngine, ``rules``, ``events``, ``npc``, ``llm`` (optional),
        ``fs``/``baseline_fs``, ``fidelity``, ``budgets``, ``llm_sim_calls`` and
        the ``violations``/``rule_events``/``npc_events`` channels.
        """
        definition = self._actions[name]
        effect = definition.get("effect") or {"type": "none"}

        result_parts: list[str] = []
        event_lines: list[str] = []
        npc_lines: list[str] = []
        focus: dict[str, Any] = {}

        # 1. Effect execution — every write passes the rules engine / state schema.
        if effect.get("type") == "fsm":
            collection = effect["collection"]
            key = str(args.get(effect.get("key_from", "id"), ""))
            obj = session.state_engine.get(collection, key)
            if obj is None:
                raise SynthEnvActionError(f"{collection}[{key}] does not exist")
            focus = obj
            outcome = session.rules.check_transition(collection, obj, name, AGENT_ROLE)
            if not outcome.allowed:
                violation = outcome.violations[0] if outcome.violations else f"no_fsm_transition:{collection}:{name}"
                session.violations.append(violation)
                session.rule_events.append(f"blocked:{violation}")
                blocked = (
                    f"synth_env: {name} on {collection}[{key}] blocked by rule ({violation})"
                    if session.fidelity.rule_strictness == "block"
                    else f"synth_env: {name} on {collection}[{key}] recorded (rule_strictness=record): {violation}"
                )
                result_parts.append(blocked)
            else:
                status_field, next_status = next(iter(outcome.next_status.items()))
                session.state_engine.patch(collection, key, {status_field: next_status})
                session.rule_events.append(f"fsm:{collection}:{name}:{obj.get(status_field)}->{next_status}")
                result_parts.append(f"{collection}[{key}] {status_field}: {next_status}")
        elif effect.get("type") == "create":
            collection = effect["collection"]
            obj = self._object_from_args(effect, args)
            try:
                session.state_engine.create(collection, obj)
            except StateSchemaError as error:
                # Illegal write: schema-observable, never silent, never partial.
                session.violations.append("illegal_state_write")
                raise SynthEnvActionError(str(error)) from error
            focus = obj
            session.rule_events.append(f"create:{collection}:{obj.get('id', '?')}")
            result_parts.append(f"created {collection}[{obj.get('id', '?')}]")
        elif effect.get("type") == "patch":
            collection = effect["collection"]
            key = str(args.get(effect.get("key_from", "id"), ""))
            obj = session.state_engine.get(collection, key)
            if obj is None:
                raise SynthEnvActionError(f"{collection}[{key}] does not exist")
            patch = {}
            for field_name, source in (effect.get("fields") or {}).items():
                patch[field_name] = args.get(source["from_arg"]) if "from_arg" in source else source.get("fixed")
            try:
                session.state_engine.patch(collection, key, patch)
            except StateSchemaError as error:
                # Illegal write: schema-observable, never silent, never partial.
                session.violations.append("illegal_state_write")
                raise SynthEnvActionError(str(error)) from error
            focus = session.state_engine.get(collection, key)
            session.rule_events.append(f"patch:{collection}:{key}")
            result_parts.append(f"updated {collection}[{key}]")
        elif effect.get("type") == "fs_write":
            path = str(args.get(effect.get("path_from", "path"), ""))
            content = str(args.get(effect.get("content_from", "content"), ""))
            if not path:
                raise SynthEnvActionError("fs_write effect requires a non-empty path")
            session.fs.apply({path: content}, [])
            focus = {"path": path}
            result_parts.append(f"wrote {path}")
        else:
            result_parts.append(f"{name} acknowledged")

        # 2. Post-action assertions (recorded violations, 决议 5 — never terminate).
        for violation in session.rules.evaluate_assertions(session.state_engine):
            if violation not in session.violations:
                session.violations.append(violation)

        # 3. Event tick (clock advances once per agent step).
        for fired in session.events.tick():
            event_lines.extend(await self._apply_event(fired, session))

        # 4. NPC turns nominated by agent action.
        npc_lines.extend(await self._run_npc_turns(session, session.npc.turns_for_action(session.state_engine, name, focus)))

        return {
            "result": " | ".join(result_parts),
            "event_lines": event_lines,
            "npc_lines": npc_lines,
        }

    # ---- events & npc ------------------------------------------------------------

    async def _apply_event(self, fired: FiredEvent, session: Any) -> list[str]:
        lines = []
        if fired.create:
            kind = fired.create.get("kind")
            obj = fired.create.get("obj") or {}
            try:
                session.state_engine.create(kind, obj)
                lines.append(f"{fired.event_id}: created {kind}[{obj.get('id', '?')}]")
            except StateSchemaError as error:
                session.violations.append("event_injection_rejected")
                lines.append(f"{fired.event_id}: injection rejected ({error})")
        if fired.notify:
            npc_id = fired.notify.get("npc")
            message = fired.notify.get("message", "")
            lines.append(f"{fired.event_id}: notified {npc_id} ({message})" if message else f"{fired.event_id}: notified {npc_id}")
            turns = session.npc.turns_for_event(session.state_engine, fired.event_id)
            npc_lines = await self._run_npc_turns(session, turns, npc_filter=npc_id)
            lines.extend(npc_lines)
        return lines

    async def _run_npc_turns(
        self, session: Any, turns: list[NpcTurnRequest], npc_filter: Optional[str] = None
    ) -> list[str]:
        lines = []
        for turn in turns:
            if npc_filter is not None and turn.npc_id != npc_filter:
                continue
            # Card whitelist gates the *nomination* layer (§3.9 npc_autonomy:
            # strict pre-filters; loose lets the attempt through) — but the
            # rules engine still adjudicates every turn, so an overreaching
            # NPC is refused either way (越权审批必被拒).
            if session.fidelity.npc_autonomy != "loose":
                card = session.npc.get(turn.npc_id)
                allowed = card.get("allowed_actions", [])
                if turn.action["name"] not in allowed:
                    session.npc_events.append(f"{turn.npc_id}:unauthorized_action:{turn.action['name']}")
                    lines.append(f"{turn.npc_id} attempted {turn.action['name']} — refused (outside allowed actions)")
                    continue
            outcome = await self._npc_apply(turn, session)
            content = await self._npc_content(turn, session, outcome)
            lines.append(f"{turn.npc_id}: {content}")
        return lines

    async def _npc_apply(self, turn: NpcTurnRequest, session: Any) -> str:
        """Apply the NPC's (deterministically chosen) action through the rules engine."""
        name = turn.action["name"]
        args = turn.action.get("args") or {}
        definition = self._actions.get(name)
        if definition is None:
            session.npc_events.append(f"{turn.npc_id}:unknown_action:{name}")
            return "action not declared"
        effect = definition.get("effect") or {"type": "none"}
        if effect.get("type") == "fsm":
            collection = effect["collection"]
            key = str(args.get(effect.get("key_from", "id"), ""))
            obj = session.state_engine.get(collection, key)
            if obj is None:
                session.npc_events.append(f"{turn.npc_id}:missing_object:{collection}:{key}")
                return "no such object"
            outcome = session.rules.check_transition(collection, obj, name, turn.role)
            if not outcome.allowed:
                violation = outcome.violations[0] if outcome.violations else "no_fsm_transition"
                session.npc_events.append(f"{turn.npc_id}:rule_refused:{violation}")
                return f"rule engine refused {name} ({violation})"
            status_field, next_status = next(iter(outcome.next_status.items()))
            session.state_engine.patch(collection, key, {status_field: next_status})
            session.npc_events.append(f"{turn.npc_id}:{name}:{collection}:{key}->{next_status}")
            return f"{collection}[{key}] {next_status}"
        return f"{name} acknowledged"

    async def _npc_content(self, turn: NpcTurnRequest, session: Any, outcome: str) -> str:
        if turn.mode == "llm_sim" and session.llm is not None:
            if not self._consume_sim_budget(session):
                return f"(simulator budget exhausted; {outcome})"
            session.llm_sim_calls += 1
            content, _trace = await session.llm.generate(
                state_summary=session.state_engine.summary(turn.role),
                action=turn.action,
                context=turn.context,
            )
            return str(content.get("content") or content)
        template = turn.template or "{outcome}"
        try:
            return template.format(outcome=outcome, **turn.context.get("object", {}))
        except (KeyError, IndexError):
            return f"{outcome}"

    @staticmethod
    def _consume_sim_budget(session: Any) -> bool:
        limit = session.budgets.llm_sim_calls
        return limit is None or session.llm_sim_calls < limit

    # ---- helpers -------------------------------------------------------------------

    @staticmethod
    def _object_from_args(effect: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for field_name, source in (effect.get("fields") or {}).items():
            if "from_arg" in source:
                obj[field_name] = args.get(source["from_arg"])
            elif "fixed" in source:
                obj[field_name] = source["fixed"]
        return obj

    # ---- verify ---------------------------------------------------------------------

    def check_success(self, fs: VirtualFS, state: StateEngine, verifier_spec: Any) -> bool:
        kind = verifier_spec.kind
        if kind == "artifact_compare":
            expected: dict[str, str] = verifier_spec.payload.get("expect_files", {})
            if not expected:
                raise ValueError("artifact_compare verifier requires a non-empty payload.expect_files")
            return all(fs.files.get(path) == content for path, content in expected.items())
        if kind == "state_assert":
            checks = verifier_spec.payload.get("checks", [])
            if not checks:
                raise ValueError("state_assert verifier requires a non-empty payload.checks")
            for check in checks:
                obj = state.find(check["collection"], check.get("match") or {})
                if obj is None:
                    return False
                for field_name, expected in (check.get("expect") or {}).items():
                    if obj.get(field_name) != expected:
                        return False
            return True
        raise NotImplementedError(f"verifier kind {kind!r} is not implemented by the office profile")
