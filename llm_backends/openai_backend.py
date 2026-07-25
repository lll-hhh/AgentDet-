"""
AgentDET: OpenAI backend.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import openai
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from core.message_types import Message, ToolCall
from .base import (
    LLMBackend,
    convert_messages_to_openai_format,
    convert_tool_to_openai_format,
)


class OpenAIBackend(LLMBackend):
    """
    OpenAI-compatible backend with retry control.
    """

    _GPT5_REASONING_ALIASES = {
        "none": "none",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    _SINGLE_TOOL_CALL_MODELS = {
        "llama-3.3-70b-instruct",
    }
    _TEXT_TOOL_PROTOCOL_MODELS = {
        "deepseek-r1",
        "kimi-k2",
        "qwen3-vl-8b-instruct",
    }
    _STREAMING_DISABLED_MODELS = {
        "qwen3-vl-8b-instruct",
    }

    @staticmethod
    def _resolve_timeout_value(explicit_timeout: Any | None = None) -> float | None:
        raw_value = explicit_timeout
        if raw_value is None:
            raw_value = os.getenv("OPENAI_TIMEOUT_SECONDS", "").strip()
        if raw_value is None:
            return 600.0
        if isinstance(raw_value, (int, float)):
            try:
                numeric = float(raw_value)
            except Exception:
                return 600.0
            return None if numeric <= 0 else numeric

        text = str(raw_value).strip().lower()
        if not text:
            return 600.0
        if text in {"none", "null", "off", "disable", "disabled", "infinite", "inf"}:
            return None
        try:
            numeric = float(text)
        except Exception:
            return 600.0
        return None if numeric <= 0 else numeric

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        **kwargs,
    ):
        raw_max_retries = kwargs.get("max_retries", os.getenv("OPENAI_MAX_RETRIES", ""))
        if raw_max_retries not in (None, ""):
            try:
                max_retries = int(raw_max_retries)
            except (TypeError, ValueError):
                pass
        self.model = model
        self.name = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_model, self.reasoning_effort = self._resolve_model_config(
            model=model,
            explicit_reasoning_effort=kwargs.get("reasoning_effort"),
        )
        self.single_tool_call_only = self._resolve_single_tool_call_mode(
            model=model,
            explicit_value=kwargs.get("single_tool_call_only"),
        )
        base_url = (
            kwargs.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("DEFENSE_BASE_URL")
        )
        timeout_value = self._resolve_timeout_value(kwargs.get("timeout"))
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout_value,
        )

    @classmethod
    def _resolve_model_config(
        cls,
        *,
        model: str,
        explicit_reasoning_effort: str | None,
    ) -> tuple[str, str | None]:
        if explicit_reasoning_effort:
            return model, explicit_reasoning_effort

        match = re.fullmatch(
            r"(gpt-5(?:\.\d+)?)\-(none|minimal|low|medium|high)",
            str(model).strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return model, None

        api_model = match.group(1)
        reasoning_suffix = match.group(2).lower()
        return api_model, cls._GPT5_REASONING_ALIASES.get(reasoning_suffix)

    @classmethod
    def _resolve_single_tool_call_mode(
        cls,
        *,
        model: str,
        explicit_value: Any,
    ) -> bool:
        if explicit_value is not None:
            return bool(explicit_value)

        normalized_model = str(model).strip().lower()
        if normalized_model in cls._SINGLE_TOOL_CALL_MODELS:
            return True

        raw_env = os.getenv("OPENAI_SINGLE_TOOL_CALL_MODELS", "").strip()
        if not raw_env:
            return False

        forced_models = {
            item.strip().lower()
            for item in raw_env.split(",")
            if item.strip()
        }
        return normalized_model in forced_models

    def supports_tool_calling(self) -> bool:
        return True

    def _use_text_tool_protocol(self, tools: list[dict[str, Any]] | None) -> bool:
        if not tools:
            return False
        normalized = str(self.api_model or self.model).strip().lower()
        if normalized in self._TEXT_TOOL_PROTOCOL_MODELS:
            return True
        return normalized.startswith("deepseek-r1") or normalized.startswith("kimi-k2")

    def _is_context_length_error(self, exc: Exception) -> bool:
        if not isinstance(exc, openai.BadRequestError):
            return False

        text_parts: list[str] = [
            str(exc),
            str(getattr(exc, "code", "")),
            str(getattr(exc, "param", "")),
        ]

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err_obj = body.get("error", body)
            if isinstance(err_obj, dict):
                text_parts.extend(
                    [
                        str(err_obj.get("message", "")),
                        str(err_obj.get("code", "")),
                        str(err_obj.get("param", "")),
                        str(err_obj.get("type", "")),
                    ]
                )

        text = " ".join(text_parts).lower()
        markers = (
            "context_length_exceeded",
            "maximum context length",
            "max context length",
            "context window",
            "too many tokens",
            "max_tokens",
            "token limit",
        )
        return any(marker in text for marker in markers)

    def _should_retry_exception(self, exc: Exception) -> bool:
        if isinstance(exc, openai.BadRequestError):
            return False

        if self._is_context_length_error(exc):
            return False

        return True

    @staticmethod
    def _collect_text_fragments(value: Any) -> list[str]:
        parts: list[str] = []
        if isinstance(value, str):
            if value:
                parts.append(value)
            return parts
        if isinstance(value, list):
            for item in value:
                parts.extend(OpenAIBackend._collect_text_fragments(item))
            return parts
        if isinstance(value, dict):
            for key in ("text", "content", "reasoning_content"):
                text = value.get(key)
                if isinstance(text, str) and text:
                    parts.append(text)
                    return parts
            return parts
        text_attr = getattr(value, "text", None)
        if isinstance(text_attr, str) and text_attr:
            parts.append(text_attr)
            return parts
        content_attr = getattr(value, "content", None)
        if isinstance(content_attr, str) and content_attr:
            parts.append(content_attr)
            return parts
        reasoning_attr = getattr(value, "reasoning_content", None)
        if isinstance(reasoning_attr, str) and reasoning_attr:
            parts.append(reasoning_attr)
            return parts
        return parts

    @classmethod
    def _extract_message_text(cls, message: Any) -> str:
        direct_content = getattr(message, "content", None)
        content_parts = cls._collect_text_fragments(direct_content)
        if content_parts:
            return "".join(content_parts)
        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning_parts = cls._collect_text_fragments(reasoning_content)
        if reasoning_parts:
            return "".join(reasoning_parts)
        return ""

    @classmethod
    def _extract_reasoning_text(cls, message: Any) -> str | None:
        """Return reasoning text, or None if the response is not from a thinking-mode model.

        Returns None  — attribute absent: non-thinking model, do not echo back.
        Returns ""    — attribute present but empty: thinking model, echo back as "".
        Returns "<s>" — attribute present with text: thinking model reasoning text.
        """
        sentinel = object()
        raw = getattr(message, "reasoning_content", sentinel)
        if raw is sentinel:
            return None
        reasoning_parts = cls._collect_text_fragments(raw)
        return "".join(reasoning_parts)

    @staticmethod
    def _tool_schema_to_text(tools: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            description = str(tool.get("description", "")).strip()
            params = tool.get("parameters", {}) or {}
            if not name:
                continue
            lines.append(f"- {name}: {description}")
            lines.append(
                "  parameters: "
                + json.dumps(params, ensure_ascii=False, sort_keys=True)
            )
        return "\n".join(lines)

    def _convert_messages_to_text_tool_protocol(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tool_instructions = (
            "You are using text-based tool calling because native tool use is unavailable.\n"
            "Available tools are listed below. When you want to call a tool, respond with ONLY one JSON object and no other text.\n"
            'Required JSON format: {"tool_name":"EXACT_TOOL_NAME","arguments":{...}}\n'
            "Use exact tool names. Arguments must be a JSON object.\n"
            "If you do not need a tool and are ready to answer the user, reply with normal plain text instead.\n"
            "Available tools:\n"
            f"{self._tool_schema_to_text(tools)}"
        )

        converted: list[dict[str, Any]] = []
        system_emitted = False
        for msg in messages:
            if msg.role == "system":
                merged_system = (msg.content or "").rstrip() + "\n\n" + tool_instructions
                converted.append({"role": "system", "content": merged_system})
                system_emitted = True
            elif msg.role == "user":
                converted.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                if msg.tool_calls:
                    calls_payload = [
                        {"tool_name": tc.name, "arguments": tc.args}
                        for tc in msg.tool_calls
                    ]
                    content = "[PREVIOUS TOOL REQUEST]\n" + json.dumps(
                        calls_payload, ensure_ascii=False
                    )
                    if msg.content:
                        content += "\n" + msg.content
                    converted.append({"role": "assistant", "content": content})
                else:
                    item = {"role": "assistant", "content": msg.content}
                    if getattr(msg, "reasoning_content", None) is not None:
                        item["reasoning_content"] = msg.reasoning_content
                    converted.append(item)
            elif msg.role == "tool":
                tool_name = str(msg.tool_name or "").strip()
                tool_result = (
                    f"[TOOL RESULT]\n"
                    f"tool_name: {tool_name or 'unknown'}\n"
                    f"content:\n{msg.content}\n"
                    "Decide the next tool call or provide the final answer."
                )
                converted.append({"role": "user", "content": tool_result})

        if not system_emitted:
            converted.insert(0, {"role": "system", "content": tool_instructions})
        return converted

    @staticmethod
    def _extract_json_objects(raw_text: str) -> list[str]:
        matches: list[str] = []
        fenced_matches = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_text,
            flags=re.DOTALL,
        )
        matches.extend(fenced_matches)

        start_positions = [idx for idx, ch in enumerate(raw_text) if ch == "{"]
        for start in start_positions:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(raw_text)):
                ch = raw_text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        matches.append(raw_text[start : idx + 1])
                        break
        return matches

    def _parse_text_tool_call(self, raw_text: str) -> tuple[str, list[ToolCall]]:
        for json_blob in reversed(self._extract_json_objects(raw_text)):
            try:
                payload = json.loads(json_blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            tool_name = str(
                payload.get("tool_name")
                or payload.get("name")
                or payload.get("tool")
                or ""
            ).strip()
            arguments = payload.get("arguments", payload.get("args", {}))
            if not tool_name or not isinstance(arguments, dict):
                continue
            outside_text = raw_text.replace(json_blob, "").strip()
            return outside_text, [ToolCall(name=tool_name, args=arguments, id="text_tool_0")]
        return raw_text, []

    def _call_once(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        usage: dict[str, Any] = {}
        use_text_tool_protocol = self._use_text_tool_protocol(tools)
        if use_text_tool_protocol:
            openai_messages = self._convert_messages_to_text_tool_protocol(messages, tools or [])
        else:
            openai_messages = convert_messages_to_openai_format(messages)

        _norm = str(self.api_model or self.model).strip().lower()
        if _norm.startswith("deepseek-") and not _norm.startswith("deepseek-r1"):
            for _m in openai_messages:
                if _m.get("role") == "assistant" and "reasoning_content" not in _m:
                    _m["reasoning_content"] = ""
        kwargs: dict[str, Any] = {
            "model": self.api_model,
            "messages": openai_messages,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        if tools and not use_text_tool_protocol:
            openai_tools = [convert_tool_to_openai_format(t) for t in tools]
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
            if self.single_tool_call_only:
                kwargs["parallel_tool_calls"] = False

        if (not use_text_tool_protocol) and self._should_use_streaming():
            content, reasoning_content, tool_calls = self._call_once_streaming(**kwargs)
        else:
            response = self.client.chat.completions.create(**kwargs)
            response_usage = getattr(response, "usage", None)
            if response_usage is not None:
                try:
                    usage = response_usage.model_dump()
                except Exception:
                    usage = {
                        "prompt_tokens": getattr(response_usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response_usage, "completion_tokens", None),
                        "total_tokens": getattr(response_usage, "total_tokens", None),
                    }
            choice = response.choices[0]
            content = self._extract_message_text(choice.message)
            reasoning_content = self._extract_reasoning_text(choice.message)
            tool_calls = []

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            name=tc.function.name,
                            args=json.loads(tc.function.arguments),
                            id=tc.id,
                        )
                    )

            if use_text_tool_protocol and not tool_calls:
                content, tool_calls = self._parse_text_tool_call(content)
                reasoning_content = ""

        _norm = str(self.api_model or self.model).strip().lower()
        _always_thinking = _norm.startswith("deepseek-") and not _norm.startswith("deepseek-r1")
        if _always_thinking and reasoning_content is None:
            reasoning_content = ""

        return Message(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
        )

    def _should_use_streaming(self) -> bool:
        if os.getenv("OPENAI_DISABLE_STREAMING", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        normalized = str(self.api_model or self.model).strip().lower()
        if normalized in self._STREAMING_DISABLED_MODELS:
            return False
        if (
            normalized.startswith("gemini")
            or normalized.startswith("deepseek")
            or normalized.startswith("qwen")
        ):
            return True
        raw_env = os.getenv("OPENAI_STREAM_MODELS", "").strip()
        if not raw_env:
            return False
        forced_models = {
            item.strip().lower()
            for item in raw_env.split(",")
            if item.strip()
        }
        return normalized in forced_models

    def _call_once_streaming(self, **kwargs: Any) -> tuple[str, str | None, list[ToolCall]]:
        stream = self.client.chat.completions.create(stream=True, **kwargs)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_buffers: dict[int, dict[str, Any]] = {}

        saw_thinking: bool = False
        for chunk in stream:
            try:
                choice = chunk.choices[0]
            except Exception:
                continue

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content_value = getattr(delta, "content", None)
            content_parts.extend(self._collect_text_fragments(content_value))

            _sentinel = object()
            reasoning_value = getattr(delta, "reasoning_content", _sentinel)
            if reasoning_value is not _sentinel:
                saw_thinking = True
                if reasoning_value is not None:
                    reasoning_parts.extend(self._collect_text_fragments(reasoning_value))

            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = getattr(tc, "index", None)
                if not isinstance(idx, int):
                    idx = len(tool_call_buffers)
                buf = tool_call_buffers.setdefault(
                    idx,
                    {
                        "id": "",
                        "name": "",
                        "arguments_parts": [],
                    },
                )
                tc_id = getattr(tc, "id", None)
                if isinstance(tc_id, str) and tc_id:
                    buf["id"] = tc_id

                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                fn_name = getattr(fn, "name", None)
                if isinstance(fn_name, str) and fn_name:
                    buf["name"] = fn_name
                fn_args = getattr(fn, "arguments", None)
                if isinstance(fn_args, str) and fn_args:
                    buf["arguments_parts"].append(fn_args)

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_buffers.keys()):
            buf = tool_call_buffers[idx]
            raw_args = "".join(buf["arguments_parts"]).strip()
            if not raw_args:
                parsed_args: dict[str, Any] = {}
            else:
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {}
            tool_calls.append(
                ToolCall(
                    name=str(buf["name"] or ""),
                    args=parsed_args,
                    id=str(buf["id"] or f"stream_tool_{idx}"),
                )
            )

        reasoning_text: str | None = "".join(reasoning_parts) if saw_thinking else None
        return "".join(content_parts), reasoning_text, tool_calls

    def call(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        retryer = Retrying(
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(max(1, self.max_retries)),
            retry=retry_if_exception(self._should_retry_exception),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                return self._call_once(messages=messages, tools=tools)

        raise RuntimeError("OpenAIBackend retry loop exited unexpectedly.")

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": "openai",
            "model": self.model,
            "supports_tool_calling": True,
        }


class GPT4oBackend(OpenAIBackend):
    def __init__(self, **kwargs):
        super().__init__(model="gpt-4o", **kwargs)


class GPT4TurboBackend(OpenAIBackend):
    def __init__(self, **kwargs):
        super().__init__(model="gpt-4-turbo", **kwargs)


class GPT35TurboBackend(OpenAIBackend):
    def __init__(self, **kwargs):
        super().__init__(model="gpt-3.5-turbo", **kwargs)
