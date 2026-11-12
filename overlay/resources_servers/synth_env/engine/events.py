# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event scheduler + simulation clock (design §3.2 六件套, §3.9 旋钮).

Events are declared by the package's ``events.yaml``: deterministic arrivals at
sim-clock positions (``at`` minutes, optionally recurring every ``every``
minutes). Each agent step advances the clock by ``step_minutes ×
time_compression`` — the compression knob squeezes waiting time while
*preserving event order, not wall times* (保序不保时). ``event_injection_rate``
scales arrival density: arrivals are scheduled at ``at / rate``, so rate > 1
packs the same ordered sequence earlier.

Every event can carry a state injection (``create``) and/or an NPC notification
(``notify``). Everything here is deterministic given the package + knobs; zero
LLM on this path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FiredEvent:
    """One event occurrence, in firing order."""

    event_id: str
    at: float
    create: dict[str, Any] = field(default_factory=dict)  # {"kind": ..., "obj": {...}} or {}
    notify: dict[str, Any] = field(default_factory=dict)  # {"npc": npc_id, ...} or {}


class EventScheduler:
    """Deterministic event queue over a sim clock, driven by the host step loop."""

    def __init__(self, event_model: dict[str, Any], step_minutes: float = 15.0):
        self._events = list((event_model or {}).get("events", []))
        self._step_minutes = step_minutes
        self._compression = 1.0
        self._rate = 1.0
        self._clock = 0.0
        # Arrival cursor per declared event, in declaration order for stability.
        self._next: list[tuple[dict[str, Any], float]] = []
        for event in self._events:
            first = self._arrival(event)
            self._next.append([event, first])

    def _arrival(self, event: dict[str, Any]) -> float:
        rate = self._rate if self._rate > 0 else 1.0
        return float(event.get("at", 0.0)) / rate

    def configure(self, *, time_compression: float = 1.0, injection_rate: float = 1.0) -> None:
        """Apply the fidelity knobs (§3.9). Re-scales pending arrivals."""
        self._compression = max(float(time_compression), 1e-9)
        self._rate = float(injection_rate) if float(injection_rate) > 0 else 1.0
        for cursor in self._next:
            cursor[1] = self._arrival(cursor[0])

    @property
    def clock(self) -> float:
        return self._clock

    def tick(self) -> list[FiredEvent]:
        """Advance one step and return the events that fire, in (arrival, id) order."""
        self._clock += self._step_minutes * self._compression
        fired: list[FiredEvent] = []
        for cursor in self._next:
            event, arrival = cursor
            every = event.get("every")
            while arrival is not None and arrival <= self._clock:
                fired.append(
                    FiredEvent(
                        event_id=event.get("id", "event"),
                        at=round(arrival, 6),
                        create=event.get("create") or {},
                        notify=event.get("notify") or {},
                    )
                )
                if every is not None and float(every) > 0:
                    arrival += float(every) / self._rate
                    cursor[1] = arrival
                else:
                    cursor[1] = float("inf")
                    arrival = None
        return sorted(fired, key=lambda fired_event: (fired_event.at, fired_event.event_id))
