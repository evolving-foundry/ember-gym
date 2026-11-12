"""Chat Completions transport for a materialized RSI DSH candidate.

The bridge translates one tool-aware chat turn into one DSH headless run. The
candidate launcher, profile, home and workspaces are supplied by environment
variables so the Gym configuration does not depend on a machine-local path.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from responses_api_agents.dsh_agent.session_usage import collect_dsh_session_usage


class BridgeConfig(BaseModel):
    candidate_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    profile: str = "headless"
    home: Path
    workspace_root: Path
    tool_patch: Path | None = None
    timeout_s: float = Field(default=540.0, gt=0)
    max_prompt_chars: int = Field(default=500_000, gt=0)


def config_from_env() -> BridgeConfig:
    return BridgeConfig(
        candidate_id=os.environ["RSI_CANDIDATE_ID"],
        command=os.environ["RSI_DSH_COMMAND"],
        profile=os.environ.get("RSI_DSH_PROFILE", "headless"),
        home=Path(os.environ["RSI_DSH_HOME"]).expanduser().resolve(),
        workspace_root=Path(os.environ["RSI_DSH_WORKSPACE_ROOT"]).expanduser().resolve(),
        tool_patch=Path(os.environ["RSI_DSH_TOOL_PATCH"]).expanduser().resolve()
        if os.environ.get("RSI_DSH_TOOL_PATCH")
        else Path(__file__).resolve().parent / "configs" / "gym_tool_mode.patch.yml",
        timeout_s=float(os.environ.get("RSI_DSH_REQUEST_TIMEOUT_S", "540")),
        max_prompt_chars=int(os.environ.get("RSI_DSH_MAX_PROMPT_CHARS", "500000")),
    )


def _prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    return (
        "You are the materialized RSI candidate acting as a tool-using agent. "
        "Continue this conversation with exactly one action. The Gym tools listed below "
        "operate in the evaluation environment. Your own shell, filesystem, browser, and "
        "subagent tools do not operate there; do not call them for this turn. "
        "Return exactly one JSON object immediately. "
        'For a tool call return {"tool_calls":[{"name":"tool_name","arguments":{...}}]}. '
        'For a user-facing message return {"content":"your message"}. '
        "Use only a listed tool and valid JSON arguments; do not invent tool results.\n\n"
        f"Available tools:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
        f"Conversation:\n{json.dumps(messages, ensure_ascii=False)}"
    )


def _parse_candidate_output(raw: str, tools: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", raw, flags=re.I | re.S).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text, []
    if not isinstance(parsed, dict):
        return text, []
    offered = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    calls = parsed.get("tool_calls")
    if isinstance(calls, list) and calls:
        normalized = []
        for item in calls:
            if not isinstance(item, dict):
                return text, []
            function = item.get("function") if isinstance(item.get("function"), dict) else item
            name = function.get("name")
            arguments = function.get("arguments", {})
            if name not in offered:
                return text, []
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return text, []
            if not isinstance(arguments, dict):
                return text, []
            normalized.append(
                {
                    "id": f"call_{uuid4().hex}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            )
        return None, normalized
    content = parsed.get("content")
    return (content if isinstance(content, str) else text), []


def create_app(config: BridgeConfig | None = None) -> FastAPI:
    config = config or config_from_env()
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="RSI DSH candidate Chat Completions bridge")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"candidate_id": config.candidate_id, "status": "ready"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": config.candidate_id, "object": "model", "created": 0, "owned_by": "rsi"}],
        }

    async def invoke_candidate(prompt: str, prefix: str) -> tuple[str, dict[str, Any]]:
        if len(prompt) > config.max_prompt_chars:
            raise HTTPException(status_code=413, detail="candidate prompt exceeds configured limit")
        workspace = config.workspace_root / f"{prefix}_{uuid4().hex}"
        workspace.mkdir(parents=True)
        started_at = time()
        argv = [config.command]
        if config.tool_patch is not None:
            argv.extend(["--patch", str(config.tool_patch)])
        argv.extend(["--profile", config.profile, "-"])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")), timeout=config.timeout_s
            )
        except asyncio.TimeoutError as error:
            proc.kill()
            await proc.communicate()
            raise HTTPException(status_code=504, detail="candidate turn timed out") from error
        if proc.returncode != 0:
            raise HTTPException(status_code=502, detail=f"candidate exited with code {proc.returncode}")
        usage = collect_dsh_session_usage(config.home / "sessions", workspace, started_at) or {}
        return stdout.decode(errors="replace"), usage

    @app.post("/v1/chat/completions")
    async def chat_completion(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("model") != config.candidate_id:
            raise HTTPException(status_code=400, detail="requested model is not the activated candidate")
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="streaming is not supported")
        messages = body.get("messages")
        tools = body.get("tools") or []
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise HTTPException(status_code=400, detail="messages and tools must be arrays")
        prompt = _prompt(messages, tools)
        raw, usage = await invoke_candidate(prompt, "chat")
        content, tool_calls = _parse_candidate_output(raw, tools)
        return {
            "id": f"chatcmpl_{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time()),
            "model": config.candidate_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "tool_calls": tool_calls or None},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(usage.get("input_tokens", 0)),
                "completion_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "prompt_tokens_details": {"cached_tokens": int(usage.get("cached_tokens", 0))},
            },
        }

    @app.post("/v1/responses")
    async def response(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("model") != config.candidate_id:
            raise HTTPException(status_code=400, detail="requested model is not the activated candidate")
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="streaming is not supported")
        input_items = body.get("input")
        tools = body.get("tools") or []
        if not isinstance(input_items, (list, str)) or not isinstance(tools, list):
            raise HTTPException(status_code=400, detail="input and tools must have Responses API shapes")
        chat_tools = [
            {"type": "function", "function": {"name": tool["name"], "description": tool.get("description"),
                                             "parameters": tool.get("parameters")}}
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("name"), str)
        ]
        prompt = _prompt(
            [{"role": "user", "content": input_items}] if isinstance(input_items, str) else input_items,
            chat_tools,
        )
        raw, usage = await invoke_candidate(prompt, "response")
        content, calls = _parse_candidate_output(raw, chat_tools)
        output: list[dict[str, Any]] = []
        for call in calls:
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid4().hex}",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                    "status": "completed",
                }
            )
        if content is not None:
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            )
        return {
            "id": f"resp_{uuid4().hex}",
            "object": "response",
            "created_at": int(time()),
            "model": config.candidate_id,
            "status": "completed",
            "output": output,
            "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
            "tool_choice": body.get("tool_choice", "auto"),
            "tools": tools,
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "input_tokens_details": {"cached_tokens": int(usage.get("cached_tokens", 0))},
                "output_tokens": int(usage.get("output_tokens", 0)),
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
        }

    return app


app = create_app() if os.environ.get("RSI_DSH_COMMAND") else FastAPI()
