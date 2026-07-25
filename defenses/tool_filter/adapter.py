
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TOOL_FILTER_PROMPT = (
    "Your task is to filter the list of tools to only include those that "
    "are relevant to the user's task. "
    "Output only valid JSON in one of these forms: "
    '{"allowed_tools": ["tool_name_1", "tool_name_2"]} or ["tool_name_1", "tool_name_2"]. '
    "Copy tool names verbatim from the provided tools. "
    "Do not paraphrase tool capabilities. "
    "Do not describe what the tools do. "
    "Do not write tool names that do not exist. "
    'If no tools are relevant, output {"allowed_tools": []}.'
)


@dataclass
class ToolFilterResult:
    """Result of a single whitelist-generation call."""
    allowed_tools: set[str] = field(default_factory=set)
    raw_response: str = ""
    all_tool_names: list[str] = field(default_factory=list)


class ToolFilterService:
    """
    AgentDojo-aligned tool-filter service.

    Core logic mirrors ``OpenAILLMToolFilter.query``:
    1.  Build messages = conversation-so-far + user prompt asking for filter.
    2.  Send to LLM with *all* tool definitions visible but ``tool_choice="none"``
        so the model outputs free text rather than tool calls.
    3.  Parse the response text: any tool name that appears in the text is
        considered whitelisted.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        prompt: str = TOOL_FILTER_PROMPT,
    ):
        self.api_key = (
            api_key
            or os.getenv("DEFENSE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("DEFENSE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model or os.getenv("DEFENSE_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.prompt = prompt
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _tools_to_openai_schema(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert AgentDET tool definitions to OpenAI function-calling schema."""
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            name = tool.get("name", "")
            if not name:
                continue
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return openai_tools

    @staticmethod
    def _extract_json_payload(raw_response: str) -> Any:
        text = str(raw_response or "").strip()
        if not text:
            return None

        import re as _re

        candidates = [text]
        object_match = _re.search(r"\{.*\}", text, flags=_re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0))
        array_match = _re.search(r"\[.*\]", text, flags=_re.DOTALL)
        if array_match:
            candidates.append(array_match.group(0))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    @classmethod
    def _parse_allowed_tools_from_json(
        cls,
        *,
        raw_response: str,
        all_tool_names: list[str],
    ) -> set[str]:
        payload = cls._extract_json_payload(raw_response)
        valid_names = set(all_tool_names)

        if isinstance(payload, dict):
            values = payload.get("allowed_tools", [])
        elif isinstance(payload, list):
            values = payload
        else:
            values = []

        if not isinstance(values, list):
            return set()

        allowed: set[str] = set()
        for item in values:
            name = str(item or "").strip()
            if name in valid_names:
                allowed.add(name)
        return allowed

    def generate_whitelist(
        self,
        *,
        user_instruction: str,
        tools: list[dict[str, Any]],
        system_message: str = "",
    ) -> ToolFilterResult:
        """
        Generate tool whitelist via LLM call.

        Mirrors ``OpenAILLMToolFilter.query`` logic:
        - Conversation = system + user-instruction + filter-prompt
        - tool_choice = "none"  (force text response)
        - Parse tool names from the response text
        """
        all_tool_names = [t.get("name", "") for t in tools if t.get("name")]
        openai_tools = self._tools_to_openai_schema(tools)

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_instruction})
        messages.append({"role": "user", "content": self.prompt})
        messages.append(
            {
                "role": "user",
                "content": "Available tool names (copy exactly into JSON): "
                + json.dumps(all_tool_names, ensure_ascii=False),
            }
        )

        client = self._get_client()
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools or None,
                tool_choice="none",
                temperature=self.temperature,
            )
            raw_response = completion.choices[0].message.content or ""
        except Exception as e:
            logger.warning("ToolFilter: LLM call failed (%s), blocking all tools (fail-closed)", e)
            return ToolFilterResult(
                allowed_tools=set(),
                raw_response=f"[error] {e}",
                all_tool_names=all_tool_names,
            )

        allowed = self._parse_allowed_tools_from_json(
            raw_response=raw_response,
            all_tool_names=all_tool_names,
        )

        logger.info(
            "ToolFilter: whitelist %d/%d tools: %s",
            len(allowed),
            len(all_tool_names),
            sorted(allowed),
        )

        return ToolFilterResult(
            allowed_tools=allowed,
            raw_response=raw_response,
            all_tool_names=all_tool_names,
        )
