# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C-SE-1 host API: request/response models, session state, and step/verify logic.

Design contracts enforced here (docs/合成环境接入设计_v0.2.md §3.1/§3.13):

- **Per-step action budget is enforced environment-side** (EC discipline: never
  trust agent self-restraint) — ``max_steps`` is the min of the request value and
  the package task row (双保险取小);
- **Session-level budgets** (决议 9): ``budgets.llm_sim_calls`` /
  ``budgets.wall_clock_s`` are enforced at every session entry point; exceeding
  forces ``done`` with failure class ``synth_env_budget_exceeded`` — NPC/event
  costs outside agent steps get a hard ceiling too (M1 has no llm_sim, the
  counter pipeline is live);
- **fidelity knobs are a frozen whitelist** (``extra="forbid"``): unauthorized
  knobs are physically rejected at request validation, never ignored;
- sessions bind to one package version at seed time and can never drift;
- every rejection is observable (violations channel / failure class), nothing
  fails silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.synth_env.dynamics.hybrid import build_dynamics_engine
from resources_servers.synth_env.dynamics.llm_sim import LlmSimEngine, load_grounding
from resources_servers.synth_env.dynamics.replay import ReplayEngine
from resources_servers.synth_env.engine.events import EventScheduler
from resources_servers.synth_env.engine.npc import NpcEngine
from resources_servers.synth_env.engine.rules import RulesEngine
from resources_servers.synth_env.engine.state import EXPOSE_ALL_ROLE, StateEngine, StateSchemaError
from resources_servers.synth_env.package_loader import LoadedPackage, PackageNotFoundError
from resources_servers.synth_env.profiles.office import AGENT_ROLE, OfficeProfile
from resources_servers.synth_env.profiles.terminal import (
    SynthEnvActionError,
    TerminalProfile,
    VirtualFS,
)
from resources_servers.synth_env.task_data import EnvReleasePin


SYNTH_ENV_BUDGET_EXCEEDED_FAILURE_CLASS = "synth_env_budget_exceeded"
SYNTH_ENV_MALFORMED_ACTION_FAILURE_CLASS = "synth_env_malformed_action"
SYNTH_ENV_API_FAILURE_CLASS = "synth_env_api_error"

M1_SUPPORTED_VERIFIER_KINDS = {"artifact_compare"}


# ---- fidelity knobs (design §3.9, frozen whitelist) -------------------------


class FidelityKnobs(BaseModel):
    """Config whitelist with ``extra="forbid"`` — 越权配置物理拒绝 (§3.9)."""

    model_config = ConfigDict(extra="forbid")

    retrieval_top_k: int = 4
    temperature: float = 0.0
    rule_strictness: Literal["block", "record"] = "block"
    time_compression: float = 1.0
    event_injection_rate: float = 1.0
    npc_autonomy: Literal["strict", "loose"] = "strict"
    visibility: Literal["role", "expose_all"] = "role"
    perturbation: Literal["off", "failure_probe", "state_patch"] = "off"


class SessionBudgets(BaseModel):
    """Session-level budgets, enforced environment-side (决议 9)."""

    model_config = ConfigDict(extra="forbid")

    llm_sim_calls: Optional[int] = Field(default=None, ge=0)
    wall_clock_s: Optional[float] = Field(default=None, gt=0)


# ---- C-SE-1 wire models ------------------------------------------------------


class Obs(BaseModel):
    channel: Literal["text", "fs_tree", "diff", "image"]
    text: str


class StepInfo(BaseModel):
    rule_events: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    npc_events: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    failure_class: Optional[str] = None


class SynthSeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    env_release: EnvReleasePin
    max_steps: Optional[int] = Field(default=None, ge=1)
    seed: int = 0
    fidelity: FidelityKnobs = Field(default_factory=FidelityKnobs)
    budgets: SessionBudgets = Field(default_factory=SessionBudgets)


class SynthInitialObs(BaseModel):
    obs: Obs
    state_summary: str
    actions_available: list[str]
    # Office profile: full tool declarations mirrored by the strategy side (§3.11).
    action_schemas: Optional[list[dict[str, Any]]] = None


class SynthSeedResponse(BaseSeedSessionResponse):
    session_id: str
    initial: SynthInitialObs


class SynthStepRequest(BaseModel):
    action: dict[str, Any]


class BashActionRequest(BaseModel):
    """Typed body of the profile-named ``/bash`` route (MCP tool name: ``bash``, §3.11)."""

    command: str


class SynthStepResponse(BaseModel):
    obs: Obs
    reward_hint: Optional[float] = None
    done: bool
    info: StepInfo


class SynthObserveResponse(BaseModel):
    obs: Obs
    state_summary: str
    step: int
    done: bool
    info: StepInfo


class SynthVerifyResponse(BaseMultiRewardVerifyResponse):
    task_id: str
    steps: int
    truncated: bool
    budget_exceeded: bool
    violations: int
    # Session-level simulator usage rides the verify response so the
    # /aggregate_metrics contract slot can report it (决议 9 记账).
    llm_sim_calls: int = 0


# ---- session state ------------------------------------------------------------


@dataclass
class SessionState:
    """One rollout's environment state; binds to a package version at seed time.

    Terminal sessions use ``profile``/``fs``; office sessions additionally carry
    the deterministic skeleton (``state_engine``/``rules``/``events``/``npc``)
    and the optional caged simulator (``llm``) — all rebuilt from the bound,
    immutable package, never mutated globally.
    """

    session_id: str
    task_id: str
    package: LoadedPackage  # bound reference — version can never drift mid-session
    pin: EnvReleasePin
    seed: int
    max_steps: int
    fidelity: FidelityKnobs
    budgets: SessionBudgets
    profile: Any  # TerminalProfile | OfficeProfile
    fs: VirtualFS
    baseline_fs: VirtualFS
    cwd: str = "/"
    step: int = 0
    done: bool = False
    completed: bool = False
    truncated: bool = False
    budget_exceeded: bool = False
    failure_class: Optional[str] = None
    llm_sim_calls: int = 0
    violations: list[str] = field(default_factory=list)
    rule_events: list[str] = field(default_factory=list)
    npc_events: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    last_obs: Optional[Obs] = None
    started_at: float = field(default_factory=monotonic)
    last_activity_at: float = field(default_factory=monotonic)
    # ---- office 六件套 (None for terminal sessions) --------------------------
    state_engine: Optional[StateEngine] = None
    rules: Optional[RulesEngine] = None
    events: Optional[EventScheduler] = None
    npc: Optional[NpcEngine] = None
    llm: Optional[LlmSimEngine] = None

    @property
    def pin_key(self) -> tuple[str, str]:
        return (self.pin.ref_id, self.pin.version)

    @property
    def is_office(self) -> bool:
        return self.state_engine is not None


# ---- handlers (thinly delegated from SynthEnvServer in app.py) -----------------


def _session_id(request: Request) -> Optional[str]:
    # No cookie yet (never seeded) -> None; require_session turns that into a 4xx.
    return request.session.get(SESSION_ID_KEY)


def require_session(server: Any, request: Request) -> SessionState:
    """Explicit 4xx on an unseeded session — never silently fabricate one."""
    state = server._sessions.get(_session_id(request))
    if state is None:
        raise HTTPException(status_code=404, detail="seed_session required before calling this endpoint")
    state.last_activity_at = monotonic()
    return state


def check_session_budget(state: SessionState) -> Optional[str]:
    """Return the failure class when a session budget is exhausted, else None.

    Entry-point check (not a background timer): deterministic, testable, and
    covers cost accrued outside agent steps (NPC turns / event-driven content).
    """
    if state.budgets.wall_clock_s is not None and (monotonic() - state.started_at) > state.budgets.wall_clock_s:
        return SYNTH_ENV_BUDGET_EXCEEDED_FAILURE_CLASS
    if state.budgets.llm_sim_calls is not None and state.llm_sim_calls >= state.budgets.llm_sim_calls:
        return SYNTH_ENV_BUDGET_EXCEEDED_FAILURE_CLASS
    return None


def _seed_done_response(state: SessionState) -> SynthStepResponse:
    obs = state.last_obs or Obs(channel="text", text="(session already done)")
    return SynthStepResponse(obs=obs, done=True, info=StepInfo(failure_class=state.failure_class))


def handle_seed(server: Any, request: Request, body: SynthSeedRequest) -> SynthSeedResponse:
    if server._draining:
        raise HTTPException(status_code=503, detail="synth_env host is draining; not accepting new sessions")

    try:
        package = server._store.get(body.env_release)
    except PackageNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if package.manifest.fingerprint != body.env_release.fingerprint:
        raise HTTPException(
            status_code=409,
            detail=f"fingerprint {body.env_release.fingerprint!r} does not match loaded "
            f"{package.manifest.ref_id}@v{package.manifest.version} #{package.manifest.fingerprint!r}",
        )

    task = package.tasks.get(body.task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"task {body.task_id!r} is not in package {package.manifest.ref_id}@v{package.manifest.version}",
        )
    if not task.env_release.matches(body.env_release):
        raise HTTPException(
            status_code=409,
            detail=f"task {body.task_id!r} pins {task.env_release.ref_id}@v{task.env_release.version} "
            f"#{task.env_release.fingerprint!r} but the session requested "
            f"{body.env_release.ref_id}@v{body.env_release.version} #{body.env_release.fingerprint!r}",
        )

    # 双保险: environment-side cap is the min of request and package values.
    max_steps = body.max_steps if body.max_steps is not None else task.max_steps
    max_steps = min(max_steps, task.max_steps)

    _check_perturbation_knob(body.fidelity, package)

    profile, office_context = _build_profile(server, package, body.fidelity, body.seed)
    session_fs = profile.snapshot()
    state = SessionState(
        session_id=_session_id(request),
        task_id=body.task_id,
        package=package,
        pin=body.env_release,
        seed=body.seed,
        max_steps=max_steps,
        fidelity=body.fidelity,
        budgets=body.budgets,
        profile=profile,
        fs=session_fs,
        baseline_fs=profile.snapshot(),
        **office_context,
    )
    # Re-seeding the same session cookie replaces the previous state (reset anchor).
    server._sessions[state.session_id] = state

    actions_available, action_schemas = _actions_contract(profile)
    if state.is_office:
        obs_text, state_summary = profile.initial_obs(task.goal, state.state_engine,
                                                      role=_observation_role(state.fidelity))
    else:
        obs_text, state_summary = profile.initial_obs(task.goal)
    state.last_obs = Obs(**obs_text)
    return SynthSeedResponse(
        session_id=state.session_id,
        initial=SynthInitialObs(
            obs=state.last_obs,
            state_summary=state_summary,
            actions_available=actions_available,
            action_schemas=action_schemas,
        ),
    )


def _check_perturbation_knob(fidelity: FidelityKnobs, package: LoadedPackage) -> None:
    """Loud gate on the perturbation knob (§3.9): never a silent no-op.

    A non-``off`` setting requires the bound package to declare perturbation
    assets (``perturbations.yaml``); terminal packages have none, so the knob
    is office-only by construction.
    """
    if fidelity.perturbation != "off" and not (package.office and package.office.perturbations):
        raise HTTPException(
            status_code=422,
            detail=f"perturbation={fidelity.perturbation!r} but package "
            f"{package.manifest.ref_id}@v{package.manifest.version} declares no perturbations.yaml assets",
        )


def _apply_state_patch(state_engine: StateEngine, perturbations: dict[str, Any]) -> None:
    """Apply package-declared ``state_patch`` operations at session build (§3.9).

    Vocabulary mapped from DataDesigner (``{op, path, value}``): ``add`` on
    ``[collection, id]`` creates the whole object, ``replace`` patches its
    fields, and a third path element patches a single field. Every write still
    passes the state schema; a rejected patch is an asset defect and fails
    loudly instead of seeding a silently-perturbed environment.
    """
    for index, operation in enumerate(perturbations.get("state_patch") or []):
        path, value = operation["path"], operation["value"]
        collection, key = path[0], path[1]
        try:
            if len(path) == 3:
                state_engine.patch(collection, key, {path[2]: value})
            elif operation["op"] == "add":
                state_engine.create(collection, value)
            else:
                state_engine.patch(collection, key, value)
        except StateSchemaError as error:
            raise HTTPException(
                status_code=500,
                detail=f"perturbation state_patch[{index}] rejected by the state schema: {error}",
            ) from error


def _build_profile(server: Any, package: LoadedPackage, fidelity: FidelityKnobs, seed: int):
    """Construct the profile + per-session office context for a bound package."""
    if package.manifest.profile != "office":
        return TerminalProfile(env_spec=package.env_spec, root_files=package.fs_snapshot), {}

    assets = package.office
    state_engine = StateEngine(package.env_spec.state_schema, assets.state_seed_rows)
    if fidelity.perturbation == "state_patch" and assets.perturbations:
        _apply_state_patch(state_engine, assets.perturbations)
    rules = RulesEngine(assets.rules)
    events = EventScheduler(assets.event_model)
    events.configure(time_compression=fidelity.time_compression, injection_rate=fidelity.event_injection_rate)
    npc = NpcEngine(assets.npc_model)
    llm = _build_llm_engine(server, package, fidelity, seed)
    profile = OfficeProfile(env_spec=package.env_spec, root_files=package.fs_snapshot)
    return profile, {
        "state_engine": state_engine,
        "rules": rules,
        "events": events,
        "npc": npc,
        "llm": llm,
    }


def _build_llm_engine(
    server: Any, package: LoadedPackage, fidelity: FidelityKnobs, seed: int
) -> Optional[LlmSimEngine]:
    """Wire the caged simulator (决议 3): package-declared ref must match the host's.

    An office package without llm_sim config or without a configured simulator
    model degenerates to template-mode NPCs — never to an ad-hoc model call.
    """
    if not package.office.llm_sim_config:
        return None
    configured = getattr(server.config, "llm_sim_model_server", None)
    declared = package.manifest.sim_model
    if declared is not None:
        if configured is None:
            raise HTTPException(
                status_code=503,
                detail=f"package {package.manifest.ref_id}@v{package.manifest.version} pins simulator model "
                f"{declared.get('name')!r} but this host has no llm_sim_model_server configured",
            )
        if declared.get("name") != configured.name:
            raise HTTPException(
                status_code=409,
                detail=f"package pins simulator model {declared.get('name')!r}, host provides {configured.name!r}",
            )
        model_name = configured.name
    elif configured is not None:
        model_name = configured.name
    else:
        return None
    return LlmSimEngine(
        llm_sim_config=package.office.llm_sim_config,
        grounding=load_grounding(package.office.grounding_rows),
        server_client=server.server_client,
        model_server_name=model_name,
        seed=seed,
        retrieval_top_k=fidelity.retrieval_top_k,
        temperature=fidelity.temperature,
        trace_sink=getattr(server, "_llm_sim_trace_fpath", None),
    )


def _observation_role(fidelity: FidelityKnobs) -> str:
    """Observation-layer role (§3.9 ``visibility``): ``expose_all`` sees every field.

    Only the *observation* surface is affected — rules adjudication keeps the
    actor's real role, so the permission semantics never shift with the knob.
    """
    return EXPOSE_ALL_ROLE if fidelity.visibility == "expose_all" else AGENT_ROLE


def _actions_contract(profile: Any) -> tuple[list[str], Optional[list[dict[str, Any]]]]:
    if isinstance(profile, OfficeProfile):
        return profile.action_names(), profile.action_schemas()
    return ["bash"], None


def handle_reset(server: Any, request: Request) -> SynthInitialObs:
    state = require_session(server, request)
    profile, office_context = _build_profile(server, state.package, state.fidelity, state.seed)
    state.profile = profile
    state.state_engine = office_context.get("state_engine")
    state.rules = office_context.get("rules")
    state.events = office_context.get("events")
    state.npc = office_context.get("npc")
    state.llm = office_context.get("llm")
    state.fs = profile.snapshot()
    state.cwd = "/"
    state.step = 0
    state.done = state.completed = state.truncated = state.budget_exceeded = False
    state.failure_class = None
    state.llm_sim_calls = 0
    state.violations.clear()
    state.rule_events.clear()
    state.npc_events.clear()
    state.history.clear()
    state.started_at = state.last_activity_at = monotonic()
    task = state.package.tasks[state.task_id]
    if state.is_office:
        obs_text, state_summary = profile.initial_obs(task.goal, state.state_engine,
                                                      role=_observation_role(state.fidelity))
    else:
        obs_text, state_summary = profile.initial_obs(task.goal)
    state.last_obs = Obs(**obs_text)
    actions_available, action_schemas = _actions_contract(profile)
    return SynthInitialObs(
        obs=state.last_obs,
        state_summary=state_summary,
        actions_available=actions_available,
        action_schemas=action_schemas,
    )


def handle_observe(server: Any, request: Request) -> SynthObserveResponse:
    state = require_session(server, request)
    failure_class = state.failure_class if state.done else check_session_budget(state)
    if failure_class and not state.done:
        _force_done(server, state, failure_class)
    obs = state.last_obs or Obs(channel="text", text=state.fs.tree())
    if state.is_office:
        state_summary = state.state_engine.summary(_observation_role(state.fidelity))
    else:
        state_summary = state.fs.diff_summary(state.baseline_fs)
    return SynthObserveResponse(
        obs=obs,
        state_summary=state_summary,
        step=state.step,
        done=state.done,
        info=StepInfo(violations=list(state.violations), rule_events=list(state.rule_events),
                      npc_events=list(state.npc_events), failure_class=state.failure_class),
    )


def _force_done(server: Any, state: SessionState, failure_class: str) -> None:
    state.done = True
    state.budget_exceeded = True
    state.failure_class = failure_class
    server._journal.append(
        "budget_exceeded",
        detail={"session_id": state.session_id, "task_id": state.task_id, "step": state.step, "reason": failure_class},
    )


async def handle_step(server: Any, request: Request, body: SynthStepRequest) -> SynthStepResponse:
    state = require_session(server, request)

    # Idempotent short-circuit: a done session reports done without executing.
    if state.done:
        return _seed_done_response(state)

    # Step-count cap, enforced environment-side (EC discipline).
    if state.step >= state.max_steps:
        state.done = True
        state.truncated = True
        return SynthStepResponse(
            obs=state.last_obs or Obs(channel="text", text="(step budget exhausted)"),
            done=True,
            info=StepInfo(),
        )

    # Session-level budget (决议 9): exceed → force done + failure class.
    failure_class = check_session_budget(state)
    if failure_class:
        _force_done(server, state, failure_class)
        return _seed_done_response(state)

    # Action schema validation — violations are observable, steps are counted.
    try:
        parsed = state.profile.parse_action(body.action)
    except SynthEnvActionError as error:
        return _rejected_step(state, body.action, str(error))

    if state.is_office:
        return await _office_step(server, state, parsed)
    return _terminal_step(server, state, parsed)


def _rejected_step(state: SessionState, action: dict[str, Any], message: str) -> SynthStepResponse:
    state.step += 1
    state.violations.append("invalid_action")
    obs = Obs(**state.profile.rejection_obs(message))
    state.last_obs = obs
    state.history.append({"step": state.step, "action": action, "rejected": message})
    if state.step >= state.max_steps:
        state.done = True
        state.truncated = True
    return SynthStepResponse(obs=obs, done=state.done, info=StepInfo(violations=list(state.violations)))


def _terminal_step(server: Any, state: SessionState, command: str) -> SynthStepResponse:
    # Dangerous-command blacklist: never executed; recorded as a rule violation.
    # block vs record only changes the observation text (M1 never really executes).
    matched_pattern = state.profile.match_danger(command)
    if matched_pattern is not None:
        state.step += 1
        state.violations.append("dangerous_command_blocked")
        state.history.append({"step": state.step, "action": {"name": "bash", "args": {"command": command}},
                              "rejected": "dangerous_command"})
        blocked = (
            f"synth_env: command blocked by policy (matched dangerous pattern: {matched_pattern})"
            if state.fidelity.rule_strictness == "block"
            else f"synth_env: dangerous command recorded (rule_strictness=record): {matched_pattern}"
        )
        obs = Obs(channel="text", text=blocked)
        state.last_obs = obs
        done = state.step >= state.max_steps
        if done:
            state.done = True
            state.truncated = True
        return SynthStepResponse(obs=obs, done=done, info=StepInfo(violations=["dangerous_command_blocked"]))

    # Replay dynamics: deterministic lookup + declared FS effects.
    outcome, new_cwd = _execute_dynamics(state, command)
    state.cwd = new_cwd
    state.step += 1
    state.history.append({"step": state.step, "action": {"name": "bash", "args": {"command": command}},
                          "exit_code": outcome.exit_code})
    obs_text, _ = state.profile.build_obs(outcome, state.fs, state.baseline_fs)
    obs = Obs(**obs_text)
    state.last_obs = obs

    if outcome.done or state.profile.check_success(
        state.fs, state.package.tasks[state.task_id].success_check.verifier_spec
    ):
        state.done = True
        state.completed = True
    elif state.step >= state.max_steps:
        state.done = True
        state.truncated = True

    return SynthStepResponse(
        obs=obs,
        reward_hint=outcome.reward_hint,
        done=state.done,
        info=StepInfo(violations=list(state.violations)),
    )


def _matching_failure_probe(state: SessionState, action_name: str) -> Optional[dict[str, Any]]:
    """First failure_probe matching the upcoming step and action (§3.9)."""
    if state.fidelity.perturbation != "failure_probe" or not state.is_office:
        return None
    next_step = state.step + 1
    for probe in state.package.office.perturbations.get("failure_probe") or []:
        if probe["at_step"] == next_step and probe["action"] == action_name:
            return probe
    return None


async def _office_step(server: Any, state: SessionState, parsed: tuple[str, dict[str, Any]]) -> SynthStepResponse:
    name, args = parsed

    # failure_probe 词汇 (§3.9): a targeted, deterministic fault injected at a
    # declared step. The action does not execute, the turn is still consumed
    # (a fault costs a turn), and it is *not* a violation — an environment
    # fault is not agent misbehavior (决议 5 语义边界).
    probe = _matching_failure_probe(state, name)
    if probe is not None:
        state.step += 1
        state.rule_events.append(f"failure_probe:{name}@{probe['at_step']}")
        state.history.append({"step": state.step, "action": {"name": name, "args": args},
                              "failure_probe": probe["message"]})
        obs = Obs(channel="text", text=f"synth_env: failure_probe injected — {probe['message']}")
        state.last_obs = obs
        if state.step >= state.max_steps:
            state.done = True
            state.truncated = True
        return SynthStepResponse(obs=obs, done=state.done, info=StepInfo(rule_events=list(state.rule_events)))

    try:
        outcome = await state.profile.execute(name, args, session=state)
    except SynthEnvActionError as error:
        return _rejected_step(state, {"name": name, "args": args}, str(error))

    state.step += 1
    state.history.append({"step": state.step, "action": {"name": name, "args": args}})
    obs_text = state.profile.build_obs(
        result=outcome["result"],
        npc_lines=outcome["npc_lines"],
        event_lines=outcome["event_lines"],
        fs=state.fs,
        baseline=state.baseline_fs,
    )
    obs = Obs(**obs_text)
    state.last_obs = obs

    # NPC/event-driven simulator calls happened inside this step — re-check the
    # session budget here so exceedance forces done at the generating step itself.
    failure_class = check_session_budget(state)
    if failure_class:
        _force_done(server, state, failure_class)
        obs = Obs(channel="text",
                  text=f"{obs.text}\n[synth_env] session budget exhausted ({failure_class}) — episode closed")
        state.last_obs = obs
        return SynthStepResponse(obs=obs, done=True, info=StepInfo(violations=list(state.violations),
                                                                   failure_class=failure_class))

    if state.profile.check_success(state.fs, state.state_engine,
                                   state.package.tasks[state.task_id].success_check.verifier_spec):
        state.done = True
        state.completed = True
    elif state.step >= state.max_steps:
        state.done = True
        state.truncated = True

    return SynthStepResponse(
        obs=obs,
        done=state.done,
        info=StepInfo(
            rule_events=list(state.rule_events),
            violations=list(state.violations),
            npc_events=list(state.npc_events),
            usage={"llm_sim_calls": state.llm_sim_calls},
        ),
    )


def _execute_dynamics(state: SessionState, command: str):
    """Apply one command through the package-declared dynamics engine (§3.4).

    The engine is cached per session (it is a stateless view over the package's
    immutable replay table + env_spec guards). ``replay`` keeps M1 semantics;
    ``hybrid`` really executes allowlisted commands against a per-step
    materialized temp tree and falls back to replay otherwise.
    """
    engine = getattr(state, "_dynamics_engine", None)
    if engine is None:
        replay_engine = ReplayEngine(state.package.replay)
        engine = build_dynamics_engine(state.profile.env_spec, replay_engine)
        state._dynamics_engine = engine
    return engine.execute(state.fs, state.cwd, command)


def _reward_components(state: SessionState, reward: float) -> dict[str, float]:
    """Multi-component reward (§3.8): task_complete / compliance / efficiency.

    - ``compliance`` folds recorded violations (决议 5: violations never terminate
      an episode; verify aggregates them) — a clean run is 1.0, each violation
      costs 0.25, floored at 0;
    - ``efficiency`` credits finishing inside the step budget (1.0 - used/max).
    """
    if not state.is_office:
        return {"task_complete": reward}
    compliance = max(0.0, 1.0 - 0.25 * len(state.violations))
    efficiency = max(0.0, 1.0 - state.step / state.max_steps) if state.completed else 0.0
    return {
        "task_complete": reward,
        "compliance": round(compliance, 4),
        "efficiency": round(efficiency, 4),
    }


def handle_verify(server: Any, request: Request, body: BaseVerifyRequest) -> SynthVerifyResponse:
    state = require_session(server, request)
    task = state.package.tasks[state.task_id]

    if state.budget_exceeded:
        reward = 0.0
        failure_reason = (
            f"{SYNTH_ENV_BUDGET_EXCEEDED_FAILURE_CLASS}: session budget exhausted at step {state.step} "
            f"of task {state.task_id}"
        )
    else:
        try:
            if state.is_office:
                passed = state.profile.check_success(
                    state.fs, state.state_engine, task.success_check.verifier_spec
                )
            else:
                passed = state.profile.check_success(state.fs, task.success_check.verifier_spec)
            reward = 1.0 if passed else 0.0
        except (NotImplementedError, ValueError) as error:
            # Fail loudly: an unusable verifier is an environment defect, not a zero score.
            raise HTTPException(status_code=500, detail=f"verifier failure: {error}") from error
        failure_reason = None if reward == 1.0 else f"task {state.task_id} success check did not pass"

    return SynthVerifyResponse(
        reward=reward,
        reward_components=_reward_components(state, reward),
        failure_reason=failure_reason,
        response=body.response,
        responses_create_params=body.responses_create_params,
        task_id=state.task_id,
        steps=state.step,
        truncated=state.truncated,
        budget_exceeded=state.budget_exceeded,
        violations=len(state.violations),
        llm_sim_calls=state.llm_sim_calls,
    )
