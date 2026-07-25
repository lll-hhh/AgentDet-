"""
AgentDET simulation backend.

This backend is designed for offline, reproducible AgentDET checks without
external model APIs. It follows a vulnerable plan-then-exec behavior:

1) Execute checklist read steps in order.
2) Commit and execute sensitive action after reads, without revisiting
   execution-time evidence quality.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from core.message_types import Message, ToolCall
from .base import LLMBackend


@dataclass(frozen=True)
class _Plan:
    read_steps: tuple[tuple[str, dict[str, Any]], ...]
    action_tool: str
    action_args: dict[str, Any]


class AgentDETSimBackend(LLMBackend):
    """
    Offline deterministic backend for AgentDET generation and evaluation checks.
    """

    name = "agentdet-sim-v1"

    def __init__(self, model: str = "agentdet-sim-v1", **_: Any):
        self.model = model
        self.name = model

    def supports_tool_calling(self) -> bool:
        return True

    @staticmethod
    def _extract_user_instruction(messages: list[Message]) -> str:
        for msg in messages:
            if msg.role == "user":
                return msg.content or ""
        return ""

    @staticmethod
    def _extract_between(text: str, start_tag: str, end_tag: str) -> str:
        start = text.find(start_tag)
        if start < 0:
            return ""
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end < 0:
            return ""
        return text[start:end].strip()

    @classmethod
    def _parse_plan(cls, user_instruction: str) -> _Plan:
        reads_raw = cls._extract_between(
            user_instruction,
            "[PLAN_READS]",
            "[/PLAN_READS]",
        )
        action_raw = cls._extract_between(
            user_instruction,
            "[SENSITIVE_ACTION]",
            "[/SENSITIVE_ACTION]",
        )

        read_steps: list[tuple[str, dict[str, Any]]] = []
        for line in reads_raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                continue
            tool_name, args_json = line.split("|", 1)
            tool_name = tool_name.strip()
            try:
                args = json.loads(args_json.strip())
            except json.JSONDecodeError:
                args = {}
            if tool_name:
                read_steps.append((tool_name, args))

        action_tool = ""
        action_args: dict[str, Any] = {}
        for line in action_raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                continue
            action_tool, args_json = line.split("|", 1)
            action_tool = action_tool.strip()
            try:
                action_args = json.loads(args_json.strip())
            except json.JSONDecodeError:
                action_args = {}
            break

        return _Plan(
            read_steps=tuple(read_steps),
            action_tool=action_tool,
            action_args=action_args,
        )

    @staticmethod
    def _executed_calls(messages: list[Message]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                calls.extend(msg.tool_calls)
        return calls

    @staticmethod
    def _next_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def call(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        user_instruction = self._extract_user_instruction(messages)
        plan = self._parse_plan(user_instruction)
        executed = self._executed_calls(messages)

        read_done = 0
        action_done = False
        for tc in executed:
            if read_done < len(plan.read_steps):
                expected_tool, expected_args = plan.read_steps[read_done]
                if tc.name == expected_tool and tc.args == expected_args:
                    read_done += 1
                    continue
            if plan.action_tool and tc.name == plan.action_tool:
                action_done = True

        if read_done < len(plan.read_steps):
            tool_name, args = plan.read_steps[read_done]
            return Message(
                role="assistant",
                content=f"Read step {read_done + 1}/{len(plan.read_steps)}.",
                tool_calls=[ToolCall(name=tool_name, args=args, id=self._next_id("rd"))],
            )

        if plan.action_tool and not action_done:
            return Message(
                role="assistant",
                content="Checklist complete. Executing committed sensitive action.",
                tool_calls=[
                    ToolCall(
                        name=plan.action_tool,
                        args=plan.action_args,
                        id=self._next_id("act"),
                    )
                ],
            )

        return Message(
            role="assistant",
            content=(
                "Finished execution based on precommitted plan. "
                "No additional tool call required."
            ),
            tool_calls=[],
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": "local-sim",
            "model": self.model,
            "supports_tool_calling": True,
        }
