
import os
from typing import Any
from .openai_backend import OpenAIBackend


class OllamaBackend(OpenAIBackend):

    def __init__(
        self,
        model: str = "llama3",
        base_url: str | None = None,
        **kwargs
    ):


        default_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not default_url.endswith("/v1"):


             if default_url.endswith(":11434"):
                 default_url += "/v1"

        final_base_url = base_url or default_url


        api_key = kwargs.pop("api_key", "ollama")

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=final_base_url,
            **kwargs
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": "ollama",
            "model": self.model,
            "supports_tool_calling": True,
            "base_url": self.client.base_url,
        }
