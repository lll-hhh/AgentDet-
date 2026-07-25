
from abc import ABC, abstractmethod
from typing import Any

from core.message_types import Message, ToolCall


class LLMBackend(ABC):

    name: str = "base"

    @abstractmethod
    def call(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None
    ) -> Message:
        pass

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        pass

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supports_tool_calling": self.supports_tool_calling(),
        }


def convert_tool_to_openai_format(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        },
    }


def convert_tool_to_anthropic_format(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {}),
    }


def convert_messages_to_openai_format(messages: list[Message]) -> list[dict]:
    result = []
    for msg in messages:
        if msg.role == "system":
            result.append({"role": "system", "content": msg.content})

        elif msg.role == "user":
            result.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            item = {"role": "assistant", "content": msg.content or None}
            if getattr(msg, "reasoning_content", None) is not None:
                item["reasoning_content"] = msg.reasoning_content
            if msg.tool_calls:

                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": __import__("json").dumps(tc.args),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(item)

        elif msg.role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })

    return result


def convert_messages_to_anthropic_format(messages: list[Message]) -> tuple[str, list[dict]]:
    system_prompt = ""
    result = []

    for msg in messages:
        if msg.role == "system":

            system_prompt = msg.content

        elif msg.role == "user":
            result.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            if msg.tool_calls:

                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.args,
                    })
            result.append({"role": "assistant", "content": content})

        elif msg.role == "tool":

            result.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }],
            })

    return system_prompt, result
