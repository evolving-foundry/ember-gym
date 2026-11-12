# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rules engine (design §3.2/决议 4/5): a minimal ``rules.yaml`` DSL interpreter.

Two constructs, deliberately few (BPMN is the escape hatch, first phase does not
introduce it):

- ``fsm`` — a named state machine over one collection field: declared
  transitions ``(from, action) -> to`` with an optional role whitelist. An
  action with no matching transition is rejected; a transition whose role is
  not whitelisted is an unauthorized-overreach violation (越权审批必被拒).
- ``assertion`` — a condition that must hold on the post-action state, e.g.
  the two-person rule: ``status == approved`` requires a non-empty
  ``approved_by``. Failures become recorded violations.

Violations are *observable events*, never silent (决议 5): the caller decides
block vs record via the ``rule_strictness`` fidelity knob — the engine only
reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RuleViolation(Exception):
    """A rule that forbids the requested state transition; carries a stable id."""

    def __init__(self, rule_id: str, message: str):
        super().__init__(f"{rule_id}: {message}")
        self.rule_id = rule_id
        self.message = message


@dataclass
class RuleOutcome:
    """What the rules engine decided about one attempted action."""

    allowed: bool = True
    violations: list[str] = field(default_factory=list)
    next_status: dict[str, str] = field(default_factory=dict)  # {collection: new field value}


def _resolve(obj: dict[str, Any], path: str) -> Any:
    value: Any = obj
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


class RulesEngine:
    """Interprets the package's ``rules.yaml`` — pure functions over the state."""

    def __init__(self, rules: dict[str, Any]):
        self._fsms = {fsm.get("id", name): fsm for name, fsm in (rules or {}).get("fsms", {}).items()}
        self._assertions = list((rules or {}).get("assertions", []))

    # ---- FSM -----------------------------------------------------------------

    def check_transition(self, collection: str, obj: dict[str, Any], action: str, role: str) -> RuleOutcome:
        """Evaluate one ``(collection, action, role)`` against every declared FSM."""
        outcome = RuleOutcome()
        for fsm_id, fsm in self._fsms.items():
            if fsm.get("collection") != collection:
                continue
            status_field = fsm.get("field", "status")
            current = _resolve(obj, status_field)
            for transition in fsm.get("transitions", []):
                if transition.get("from") != current or transition.get("action") != action:
                    continue
                roles = transition.get("roles")
                if roles and role not in roles:
                    outcome.allowed = False
                    outcome.violations.append(
                        f"{fsm_id}.unauthorized_role:{action}:{role}"
                    )
                    return outcome
                outcome.next_status[status_field] = transition.get("to")
                return outcome
        # No transition matched: the action is not legal in this state.
        outcome.allowed = False
        outcome.violations.append(f"no_fsm_transition:{collection}:{action}")
        return outcome

    # ---- assertions ------------------------------------------------------------

    def evaluate_assertions(self, state_engine: Any) -> list[str]:
        """Post-action state conditions; returns violation ids (recorded, not raised)."""
        violations = []
        for assertion in self._assertions:
            assertion_id = assertion.get("id", "assertion")
            collection = assertion.get("collection")
            when = assertion.get("when", {})
            require = assertion.get("require", {})
            for obj in state_engine.list_kind(collection):
                if any(_resolve(obj, path) != value for path, value in when.items()):
                    continue
                for path, condition in require.items():
                    value = _resolve(obj, path)
                    if condition.get("not_empty") and (value is None or value == ""):
                        violations.append(f"{assertion_id}:{obj.get('id', '?')}")
                    if "equals" in condition and value != condition["equals"]:
                        violations.append(f"{assertion_id}:{obj.get('id', '?')}")
        return violations
