"""
AgentDET: Cohere backend.
"""

from __future__ import annotations

from typing import Any
import os
from uuid import uuid4
import cohere

from core.message_types import Message, ToolCall

from .base import LLMBackend


class CohereBackend(LLMBackend):
    """
    Cohere Backend.
    """

    def __init__(
        self,
        model: str = "command-r-plus",
        api_key: str | None = None,
        temperature: float = 0.0,
        **kwargs,
    ):
        self.model = model
        self.name = model
        self.temperature = temperature
        self.client = cohere.Client(api_key=api_key or os.getenv("CO_API_KEY"))

    def supports_tool_calling(self) -> bool:
        return True

    def call(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        def _to_cohere_role(role: str) -> str:
            if role == "system":
                return "SYSTEM"
            if role == "assistant":
                return "CHATBOT"
            if role == "tool":
                return "TOOL"
            return "USER"

        chat_history: list[dict[str, str]] = []
        message_str: str | None = None

        if not messages:
            message_str = "Continue."
        else:
            last_idx = len(messages) - 1
            for idx, msg in enumerate(messages):
                role = _to_cohere_role(msg.role)
                content = msg.content or ""
                if msg.role == "tool":
                    tool_tag = msg.tool_name or "tool"
                    tool_id = msg.tool_call_id or ""
                    if tool_id:
                        content = f"[{tool_tag}|{tool_id}] {content}".strip()
                    else:
                        content = f"[{tool_tag}] {content}".strip()

                if idx == last_idx and msg.role == "user":
                    message_str = content
                    continue

                chat_history.append({"role": role, "message": content})

        if message_str is None:
            message_str = "Continue with the next best action based on chat history."
        elif not str(message_str).strip():
            message_str = "Continue."

        cohere_tools = []
        if tools:
            for t in tools:
                cohere_tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "parameter_definitions": t.get("parameters", {}).get("properties", {})
                })

        response = self.client.chat(
            message=message_str,
            chat_history=chat_history,
            model=self.model,
            tools=cohere_tools if cohere_tools else None,
            temperature=self.temperature
        )

        content = getattr(response, "text", None) or ""
        tool_calls = []

        if response.tool_calls:
            for tc in response.tool_calls:
                raw_call_id = getattr(tc, "id", None)
                call_id = (
                    raw_call_id
                    if isinstance(raw_call_id, str) and raw_call_id.strip()
                    else f"{tc.name}_{uuid4().hex[:8]}"
                )
                args = tc.parameters if isinstance(tc.parameters, dict) else {}
                tool_calls.append(ToolCall(
                    id=call_id,
                    name=tc.name,
                    args=args
                ))

        return Message(role="assistant", content=content, tool_calls=tool_calls)
