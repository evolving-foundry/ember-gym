"""Build task-scoped trajectory turns from OpenCode's session database."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from nemo_gym.rollout_observability import ObservationGap, TrajectoryRecord, TrajectoryTurn


def _data(row: Any) -> dict[str, Any]:
    try:
        value = json.loads(row["data"])
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def append_opencode_turns(
    trajectory: TrajectoryRecord,
    session_ids: set[str],
    message_rows: list[Any],
    part_rows: list[Any],
) -> None:
    """Record assistant decisions and visible tool calls without guessing model IDs."""
    parts_by_message: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in part_rows:
        parts_by_message[row["message_id"]].append(_data(row))
    turn_numbers: dict[str, int] = defaultdict(int)
    step_counts: dict[str, int] = defaultdict(int)
    for row in message_rows:
        invocation_id = row["session_id"]
        message = _data(row)
        if invocation_id not in session_ids or message.get("role") != "assistant":
            continue
        turn_numbers[invocation_id] += 1
        parts = parts_by_message.get(row["id"], [])
        answers = []
        for part in parts:
            if part.get("type") == "tool":
                step_counts[invocation_id] += 1
                answers.append(
                    {
                        "type": "function_call",
                        "call_id": part.get("callID"),
                        "name": part.get("tool"),
                        "arguments": part.get("state", {}).get("input", {}),
                    }
                )
            elif part.get("type") == "text":
                answers.append({"type": "text", "text": part.get("text", "")})
        created = message.get("time", {}).get("created") or row["time_created"]
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id=invocation_id,
                task_id=trajectory.task_id,
                rollout_id=trajectory.rollout_id,
                turn_no=turn_numbers[invocation_id],
                timestamp=float(created) / 1000,
                answer=answers,
                step_count=step_counts[invocation_id],
            )
        )
    if trajectory.turns:
        trajectory.gaps.append(ObservationGap(code="turn_model_call_scope_incomplete"))


def scope_opencode_trajectory(trajectory: TrajectoryRecord, body: Any, rollout_id: str) -> dict[str, Any]:
    """Attach the current evaluation task and rollout identities to observed turns."""
    data = body.model_dump() if hasattr(body, "model_dump") else dict(body)
    task_id = str(data.get("_ng_task_index", data.get("task_id", rollout_id)))
    turns = [
        turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id})
        for turn in trajectory.turns
    ]
    scoped = trajectory.model_copy(
        update={"task_id": task_id, "rollout_id": rollout_id, "turns": turns}
    )
    return scoped.model_dump(mode="json")
