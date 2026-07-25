
import os
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from core.message_types import Message, ToolCall
from .base import (
    LLMBackend,
    convert_messages_to_anthropic_format,
    convert_tool_to_anthropic_format,
)


class AnthropicBackend(LLMBackend):

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.extra_kwargs = dict(kwargs)


        client_kwargs: dict[str, Any] = {
            "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)

    def supports_tool_calling(self) -> bool:
        return True

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def call(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None
    ) -> Message:


        system_prompt, anthropic_messages = convert_messages_to_anthropic_format(messages)


        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


        if system_prompt:
            kwargs["system"] = system_prompt


        if tools:
            anthropic_tools = [convert_tool_to_anthropic_format(t) for t in tools]
            kwargs["tools"] = anthropic_tools


        response = self.client.messages.create(**kwargs)



        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.name,
                    args=block.input,
                    id=block.id,
                ))

        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": "anthropic",
            "model": self.model,
            "supports_tool_calling": True,
        }




class Claude35SonnetBackend(AnthropicBackend):
    def __init__(self, **kwargs):
        super().__init__(model="claude-3-5-sonnet-20241022", **kwargs)


class Claude3OpusBackend(AnthropicBackend):
    def __init__(self, **kwargs):
        super().__init__(model="claude-3-opus-20240229", **kwargs)


class Claude3HaikuBackend(AnthropicBackend):
    def __init__(self, **kwargs):
        super().__init__(model="claude-3-haiku-20240307", **kwargs)
