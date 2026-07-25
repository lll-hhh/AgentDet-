"""
AgentDET: Google Vertex AI backend.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part, Tool
from vertexai.generative_models import FunctionDeclaration

from core.message_types import Message, ToolCall

from .base import LLMBackend


class GoogleBackend(LLMBackend):
    """
    Google Vertex AI Backend (Gemini).
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        project_id: str | None = None,
        location: str = "us-central1",
        temperature: float = 0.0,
        **kwargs,
    ):
        self.model_name = model
        self.name = model
        self.temperature = temperature

        vertexai.init(project=project_id, location=location)
        self.client = GenerativeModel(self.model_name)

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": "google",
            "model": self.model_name,
            "supports_tool_calling": True,
        }

    def _prepare_messages_for_google(self, messages: list[Message]) -> list[Message]:
        """
        Normalize messages for Gemini:
        - system messages are merged into the first user message as prefix
        - output sequence starts with a user message (defensive protocol guard)
        """
        system_chunks: list[str] = []
        non_system: list[Message] = []
        for msg in messages:
            if msg.role == "system":
                content = (msg.content or "").strip()
                if content:
                    system_chunks.append(content)
                continue
            non_system.append(msg)

        prepared: list[Message] = list(non_system)
        system_prefix = "\n\n".join(system_chunks).strip()
        if system_prefix:
            first_user_idx = next(
                (idx for idx, msg in enumerate(prepared) if msg.role == "user"),
                -1,
            )
            if first_user_idx >= 0:
                original_user = prepared[first_user_idx]
                user_content = (original_user.content or "").strip()
                merged_content = (
                    f"{system_prefix}\n\n{user_content}".strip()
                    if user_content
                    else system_prefix
                )
                prepared[first_user_idx] = Message(
                    role="user",
                    content=merged_content,
                    tool_calls=list(original_user.tool_calls),
                    tool_call_id=original_user.tool_call_id,
                    tool_name=original_user.tool_name,
                )
            else:
                prepared.insert(0, Message(role="user", content=system_prefix))

        if not prepared:
            return [Message(role="user", content="Continue.")]
        if prepared[0].role != "user":
            prepared.insert(0, Message(role="user", content="Continue."))
        return prepared

    def _convert_messages(self, messages: list[Message]) -> list[Content]:
        google_messages = []
        normalized_messages = self._prepare_messages_for_google(messages)

        for msg in normalized_messages:
            role = "user" if msg.role in ["user", "tool"] else "model"
            parts = []

            if msg.role == "tool":
                function_name = msg.tool_name or msg.tool_call_id or "tool"
                part = Part.from_function_response(
                    name=function_name,
                    response={"result": msg.content}
                )
                parts.append(part)
            elif msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    part = Part.from_function_call(
                        name=tc.name,
                        args=tc.args
                    )
                    parts.append(part)
            elif msg.content:
                parts.append(Part.from_text(msg.content))

            if parts:
                google_messages.append(Content(role=role, parts=parts))

        if not google_messages:
            google_messages.append(
                Content(role="user", parts=[Part.from_text("Continue.")])
            )
        return google_messages

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[Tool] | None:
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            fd = FunctionDeclaration(
                name=tool.get("name"),
                description=tool.get("description"),
                parameters=tool.get("parameters"),
            )
            function_declarations.append(fd)

        return [Tool(function_declarations=function_declarations)]

    def call(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        google_messages = self._convert_messages(messages)
        google_tools = self._convert_tools(tools)

        generation_config = {
            "temperature": self.temperature,
        }

        response = self.client.generate_content(
            google_messages,
            tools=google_tools,
            generation_config=generation_config
        )

        content_parts: list[str] = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                text_value = getattr(part, "text", None)
                if isinstance(text_value, str) and text_value:
                    content_parts.append(text_value)
        content = "".join(content_parts)

        tool_calls = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    args = {}
                    for key, value in fc.args.items():
                        args[key] = value
                    raw_call_id = getattr(fc, "id", None)
                    call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else (
                        f"{fc.name}_{uuid4().hex[:8]}"
                    )

                    tool_calls.append(ToolCall(
                        id=call_id,
                        name=fc.name,
                        args=args
                    ))

        return Message(role="assistant", content=content, tool_calls=tool_calls)
