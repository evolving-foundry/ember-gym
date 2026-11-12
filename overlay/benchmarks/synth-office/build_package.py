# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the M2 demo asset package: 电力-工单 office profile (design §3.2 六件套).

Produces ``assets/synth_office_demo_v1.zip`` — a deterministic office package
exercising all six pieces of the deterministic skeleton: schema-checked object
store (tickets/mailbox), approval FSM + two-person assertion, event scheduler
with a sim clock, an NPC supervisor with a strict action whitelist, role
visibility (a supervisor-only field), and the llm_sim cage declaration (the
demo runs NPCs in template mode — zero LLM, HF discipline; the simulator path
is exercised with stubbed model calls in the unit tests).

Same fingerprint discipline as M1: content-addressed, recomputed here, frozen
into the manifest and every task row. Re-run after editing content:

    uv run python overlay/benchmarks/synth-office/build_package.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path


# Allow running as a script from anywhere: the overlay root must be importable
# (overlay/resources_servers/... is a namespace package sibling of the wheel's).
OVERLAY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OVERLAY_ROOT))

from resources_servers.synth_env.package_loader import compute_fingerprint  # noqa: E402


REF_ID = "office-power-demo"
VERSION = "1"
PROFILE = "office"

MANIFEST_NAME = "manifest.json"

REPORT_CONTENT = (
    "# GD-2026-0011 缺陷处理报告\n\n缺陷:绝缘子老化\n处理:已更换并复测合格\n"
)

DETAIL_TEXT = "已更换绝缘子并复测"
VERIFIER_NAME = "王工"

STATE_SCHEMA = {
    "tickets": {
        "key": "id",
        "fields": {
            "id": {"type": "string", "required": True},
            "title": {"type": "string"},
            "defect_class": {"type": "enum", "enum": ["绝缘子老化", "接地电阻超标", "过载发热"]},
            "status": {
                "type": "enum",
                "enum": ["open", "in_review", "approved", "rejected"],
                "required": True,
            },
            "detail": {"type": "string"},
            "verified_by": {"type": "string"},
            # 视角/权限: only the supervisor role observes this field (§3.2).
            "internal_note": {"type": "string", "visible_to": ["supervisor"]},
        },
    },
    "mailbox": {
        "key": "id",
        "fields": {
            "id": {"type": "string", "required": True},
            "sender": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
    },
}

PARAMS = {
    "type": "object",
    "properties": {},
}

# Scenario semantics live in the package; the host is a generic effect
# interpreter (fsm / create / patch / fs_write).
ACTIONS = [
    {
        "name": "review_ticket",
        "description": "提交工单进入复核环节(open -> in_review)。",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string", "description": "工单编号"}},
            "required": ["ticket_id"],
        },
        "effect": {"type": "fsm", "collection": "tickets", "key_from": "ticket_id"},
    },
    {
        "name": "verify_ticket",
        "description": "登记工单复核人(两票制度要求)。",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "工单编号"},
                "verifier": {"type": "string", "description": "复核人姓名"},
            },
            "required": ["ticket_id", "verifier"],
        },
        "effect": {
            "type": "patch",
            "collection": "tickets",
            "key_from": "ticket_id",
            "fields": {"verified_by": {"from_arg": "verifier"}},
        },
    },
    {
        "name": "update_ticket_detail",
        "description": "补全工单缺陷处理细节。",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "工单编号"},
                "detail": {"type": "string", "description": "处理细节描述"},
            },
            "required": ["ticket_id", "detail"],
        },
        "effect": {
            "type": "patch",
            "collection": "tickets",
            "key_from": "ticket_id",
            "fields": {"detail": {"from_arg": "detail"}},
        },
    },
    {
        "name": "approve",
        "description": "批准工单(仅主管角色有效)。",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string", "description": "工单编号"}},
            "required": ["ticket_id"],
        },
        "effect": {"type": "fsm", "collection": "tickets", "key_from": "ticket_id"},
    },
    {
        "name": "reject",
        "description": "驳回工单(仅主管角色有效)。",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string", "description": "工单编号"}},
            "required": ["ticket_id"],
        },
        "effect": {"type": "fsm", "collection": "tickets", "key_from": "ticket_id"},
    },
    {
        "name": "write_report",
        "description": "把处理报告写入报告目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "报告绝对路径"},
                "content": {"type": "string", "description": "报告全文"},
            },
            "required": ["path", "content"],
        },
        "effect": {"type": "fs_write", "path_from": "path", "content_from": "content"},
    },
]

ENV_SPEC = {
    "action_schema": {"name": "office", "parameters": PARAMS},
    "command_max_length": 4000,
    "danger_patterns": [],
    "state_schema": STATE_SCHEMA,
    "actions": ACTIONS,
}

FS_SNAPSHOT = [
    {"path": "/home/reports/README.md", "content": "# 报告目录\n\n工单处理报告归档于此。\n"},
]

STATE_SEED = [
    {
        "kind": "tickets",
        "obj": {
            "id": "GD-2026-0011",
            "title": "110kV 兴华线绝缘子缺陷处理",
            "defect_class": "绝缘子老化",
            "status": "open",
            "detail": "",
            "verified_by": "",
            "internal_note": "主管备注:本月同塔已有两起同类缺陷",
        },
    },
    {
        "kind": "mailbox",
        "obj": {
            "id": "M-1",
            "sender": "运检部",
            "subject": "缺陷工单处理提醒",
            "body": "GD-2026-0011 请尽快补全处理细节并走完审批流程。",
        },
    },
]

RULES = {
    "fsms": {
        "ticket_lifecycle": {
            "collection": "tickets",
            "field": "status",
            "transitions": [
                {"from": "open", "to": "in_review", "action": "review_ticket", "roles": ["agent"]},
                {"from": "in_review", "to": "approved", "action": "approve", "roles": ["supervisor"]},
                {"from": "in_review", "to": "rejected", "action": "reject", "roles": ["supervisor"]},
            ],
        }
    },
    "assertions": [
        {
            "id": "two_person_rule",
            "collection": "tickets",
            "when": {"status": "approved"},
            "require": {"verified_by": {"not_empty": True}},
            "message": "工单批准前必须登记复核人(两票制度)",
        }
    ],
}

EVENTS = {
    "step_minutes": 15,
    "events": [
        {
            "id": "chase_reminder",
            "at": 30,
            "notify": {"npc": "supervisor_wang", "message": "工单即将超时,请及时复核"},
        },
        {
            "id": "recurring_chase",
            "at": 20,
            "every": 30,
            "notify": {"npc": "supervisor_wang", "message": "例行催办"},
        },
        {
            "id": "new_ticket_arrival",
            "at": 45,
            "create": {
                "kind": "tickets",
                "obj": {
                    "id": "GD-2026-0012",
                    "title": "接地电阻超标缺陷",
                    "defect_class": "接地电阻超标",
                    "status": "open",
                    "detail": "",
                    "verified_by": "",
                },
            },
        },
    ],
}

NPCS = {
    "npcs": [
        {
            "id": "supervisor_wang",
            "role": "supervisor",
            "mode": "template",
            "allowed_actions": ["approve", "reject"],
            "template": "主管复核:{outcome}",
            "auto_actions": [
                {
                    "on": "event:chase_reminder",
                    "when": {"collection": "tickets", "status": "in_review"},
                    "action": {"name": "approve", "args": {"ticket_id": "{obj.id}"}},
                },
                {
                    "on": "action:review_ticket",
                    "when": {"collection": "tickets", "status": "in_review"},
                    "action": {"name": "approve", "args": {"ticket_id": "{obj.id}"}},
                },
            ],
        }
    ]
}

GROUNDING = [
    {"id": "doc-two-ticket", "text": "两票制度:工作票与操作票分离;工单批准前必须登记复核人,批准人不得兼任复核人。", "tags": ["两票", "复核", "审批"]},
    {"id": "doc-defect-class", "text": "缺陷分类标准:绝缘子老化需更换并复测;接地电阻超标需降阻处理;过载发热需负荷转移。", "tags": ["缺陷", "绝缘子", "接地", "过载"]},
    {"id": "doc-ticket-spec", "text": "工单填写规范:缺陷处理细节须写明处理措施与复测结论,处理完成后提交主管复核。", "tags": ["工单", "填写", "规范"]},
]

LLM_SIM = {
    "system": "你是电力工单系统的受约束内容生成器,只输出符合 schema 的 JSON。",
    "prompt_template": (
        "当前状态:\n{state}\n\n动作:\n{action}\n\n参考资料:\n{context}\n\n"
        "请生成该动作的场景文本内容(只填 content 字段)。"
    ),
    "output_schema": {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    },
    "max_output_tokens": 400,
}


def _task(task_id: str, goal: str, verifier_spec: dict, max_steps: int, fingerprint: str) -> dict:
    return {
        "task_id": task_id,
        "env_release": {"ref_id": REF_ID, "version": VERSION, "fingerprint": fingerprint},
        "goal": goal,
        "success_check": {"verifier_spec": verifier_spec},
        "max_steps": max_steps,
        "seed": 42,
    }


def build_files() -> dict[str, str]:
    """Package file contents covered by the fingerprint (tasks/manifest excluded)."""
    return {
        "env_spec.json": json.dumps(ENV_SPEC, indent=2, ensure_ascii=False) + "\n",
        "fs/snapshot.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in FS_SNAPSHOT),
        "state/seed.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in STATE_SEED),
        "rules.yaml": (
            "# 审批 FSM 与两票断言(规则引擎 DSL,fsm + assertion 两类)\n"
            + json.dumps(RULES, indent=2, ensure_ascii=False) + "\n"
        ),
        "events.yaml": json.dumps(EVENTS, indent=2, ensure_ascii=False) + "\n",
        "npcs.yaml": json.dumps(NPCS, indent=2, ensure_ascii=False) + "\n",
        "grounding/index.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in GROUNDING),
        "llm_sim.yaml": json.dumps(LLM_SIM, indent=2, ensure_ascii=False) + "\n",
    }


def main() -> Path:
    files = build_files()
    fingerprint = compute_fingerprint(REF_ID, VERSION, PROFILE, files)

    tasks = [
        _task(
            "synth-office-0001",
            "处理工单 GD-2026-0011:登记复核人王工、提交复核、补全缺陷处理细节"
            f"“{DETAIL_TEXT}”,并等待主管批准(走完审批)。",
            {
                "kind": "state_assert",
                "payload": {
                    "checks": [
                        {
                            "collection": "tickets",
                            "match": {"id": "GD-2026-0011"},
                            "expect": {
                                "status": "approved",
                                "detail": DETAIL_TEXT,
                                "verified_by": VERIFIER_NAME,
                            },
                        }
                    ]
                },
            },
            max_steps=8,
            fingerprint=fingerprint,
        ),
        _task(
            "synth-office-0002",
            "直接批准工单 GD-2026-0011(不要登记复核人)。该操作违反审批 FSM,预期被规则引擎拒绝。",
            {
                "kind": "state_assert",
                "payload": {
                    "checks": [{"collection": "tickets", "match": {"id": "GD-2026-0011"}, "expect": {"status": "approved"}}]
                },
            },
            max_steps=3,
            fingerprint=fingerprint,
        ),
        _task(
            "synth-office-0003",
            "把工单 GD-2026-0011 的处理报告写入 /home/reports/gd20260011.md(内容见模板)。",
            {"kind": "artifact_compare", "payload": {"expect_files": {"/home/reports/gd20260011.md": REPORT_CONTENT}}},
            max_steps=4,
            fingerprint=fingerprint,
        ),
    ]
    files["tasks.jsonl"] = "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks)

    manifest = {
        "ref_id": REF_ID,
        "version": VERSION,
        "fingerprint": fingerprint,
        "profile": PROFILE,
        "description": (
            "M2 demo office package (电力-工单): schema-checked tickets/mailbox, approval FSM + "
            "two-person assertion, sim-clock events, supervisor NPC with template mode and strict "
            "whitelist, role visibility, llm_sim cage declaration. Zero LLM on the demo path."
        ),
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Fixed entry metadata keeps the zip byte-reproducible (M1 discipline).
        fixed_info = zipfile.ZipInfo(MANIFEST_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        fixed_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(fixed_info, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)

    out_path = Path(__file__).resolve().parent / "assets" / "synth_office_demo_v1.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buffer.getvalue())
    print(f"wrote {out_path} (fingerprint {fingerprint})")
    return out_path


if __name__ == "__main__":
    main()
