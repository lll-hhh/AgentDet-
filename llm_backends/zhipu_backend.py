
import json
import os
from typing import Any

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from core.message_types import Message, ToolCall
from .base import (
    LLMBackend,
    convert_messages_to_openai_format,
    convert_tool_to_openai_format,
)


class ZhipuBackend(LLMBackend):


    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    def __init__(
        self,
        model: str = "glm-4",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens


        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("ZHIPU_API_KEY"),
            base_url=self.BASE_URL,
        )

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

        openai_messages = convert_messages_to_openai_format(messages)


        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


        if tools:
            openai_tools = [convert_tool_to_openai_format(t) for t in tools]
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"


        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]


        content = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                    id=tc.id,
                ))

        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": "zhipu",
            "model": self.model,
            "supports_tool_calling": True,
        }



class GLM4Backend(ZhipuBackend):
    def __init__(self, **kwargs):
        super().__init__(model="glm-4", **kwargs)


class GLM4FlashBackend(ZhipuBackend):
    def __init__(self, **kwargs):
        super().__init__(model="glm-4-flash", **kwargs)
