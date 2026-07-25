"""
AgentDET: LLM backend factory with lazy optional imports and generic support.
"""

from __future__ import annotations

import inspect
import os
import warnings
from typing import Any

from .base import LLMBackend


MODEL_REGISTRY: dict[str, str] = {
    "gpt-4o": "openai",
    "gpt-4-turbo": "openai",
    "gpt-3.5-turbo": "openai",
    "gpt-4o-mini": "openai",
    "o1-preview": "openai",
    "o1-mini": "openai",
    "deepseek-chat": "openai",
    "deepseek-reasoner": "openai",
    "moonshot-v1-8k": "openai",
    "moonshot-v1-32k": "openai",
    "yi-large": "openai",
    "claude-3.5-sonnet": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-opus": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-haiku": "anthropic",
    "gemini-1.5-pro": "google",
    "gemini-1.5-flash": "google",
    "gemini-1.0-pro": "google",
    "command-r-plus": "cohere",
    "command-r": "cohere",
    "glm-4": "zhipu",
    "glm-4-flash": "zhipu",
    "glm-4-air": "zhipu",
    "glm-4-plus": "zhipu",
    "ollama": "ollama",
    "llama3": "ollama",
    "mistral": "ollama",
    "qwen2.5": "ollama",
    "agentdet-sim-v1": "agentdet_sim",
}


BACKENDS: dict[str, str] = MODEL_REGISTRY.copy()


MODEL_ALIASES: dict[str, str] = {
    "gpt4": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt-4": "gpt-4o",
    "o1": "o1-preview",
    "claude": "claude-3-5-sonnet-20241022",
    "claude-3.5": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "sonnet": "claude-3-5-sonnet-20241022",
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-pro",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
    "command-r+": "command-r-plus",
    "glm": "glm-4",
    "glm4": "glm-4",
    "zhipu": "glm-4",
    "llama3": "llama3",
    "mistral": "mistral",
    "qwen": "qwen2.5",
    "qwen2.5": "qwen2.5",
    "agentdet_sim": "agentdet-sim-v1",
    "agentdet-sim": "agentdet-sim-v1",
}


def normalize_model_name(name: str) -> str:
    """Normalize model aliases to canonical names when available."""
    name_lower = name.lower().strip()
    return MODEL_ALIASES.get(name_lower, name)


def _load_backend_class(provider: str):
    """Load backend classes lazily to avoid import-time hard dependency failures."""
    if provider == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend

    if provider == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend

    if provider == "google":
        from .google_backend import GoogleBackend
        return GoogleBackend

    if provider == "cohere":
        from .cohere_backend import CohereBackend
        return CohereBackend

    if provider == "zhipu":
        from .zhipu_backend import ZhipuBackend
        return ZhipuBackend

    if provider == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend

    if provider == "agentdet_sim":
        from .agentdet_sim_backend import AgentDETSimBackend
        return AgentDETSimBackend

    raise ValueError(f"Unknown backend provider: {provider}")


def _resolve_provider(normalized_name: str, kwargs: dict[str, Any] = None) -> str:
    """
    Resolve provider from model name or explicit configuration.
    Priority:
    1. Explicit 'provider' in kwargs
    2. Explicit 'base_url' -> prefer generic OpenAI-compatible routing
    3. Known prefixes/registry
    """
    if kwargs and "provider" in kwargs:
        return kwargs["provider"]

    if kwargs and kwargs.get("base_url"):
        return "openai"
    if os.getenv("OPENAI_BASE_URL") or os.getenv("DEFENSE_BASE_URL"):
        return "openai"

    if normalized_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized_name]

    if normalized_name.startswith(("gpt-", "o1-", "deepseek-", "moonshot-", "yi-")):
        return "openai"
    if normalized_name.startswith("claude-"):
        return "anthropic"
    if normalized_name.startswith(("gemini-", "goog-")):
        return "google"
    if normalized_name.startswith("command-"):
        return "cohere"
    if normalized_name.startswith("glm-"):
        return "zhipu"
    if normalized_name.startswith("qwen3-"):
        return "openai"
    if (
        "ollama" in normalized_name
        or normalized_name in ["llama3", "mistral", "qwen2.5"]
        or normalized_name.startswith("qwen")
        or ":" in normalized_name
    ):
        return "ollama"
    if normalized_name == "agentdet-sim-v1":
        return "agentdet_sim"

    return ""


def _build_missing_dependency_message(model: str, exc: ModuleNotFoundError) -> str:
    package = exc.name or "required package"
    return (
        f"Cannot initialize backend for model '{model}' because dependency '{package}' "
        "is not installed. Install optional backend dependencies from requirements.txt."
    )


def _filter_backend_kwargs(backend_class: type, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Keep only init kwargs accepted by backend __init__, unless backend accepts **kwargs.
    """
    signature = inspect.signature(backend_class.__init__)
    params = signature.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kwargs:
        return dict(kwargs), []

    filtered: dict[str, Any] = {}
    accepted_names: set[str] = set()
    for name, param in params.items():
        if name in {"self", "model"}:
            continue
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        accepted_names.add(name)
        if name in kwargs:
            filtered[name] = kwargs[name]
    dropped = sorted(k for k in kwargs.keys() if k not in accepted_names)
    return filtered, dropped


def create_llm_backend(name: str, **kwargs: Any) -> LLMBackend:
    """
    Create an LLM backend by name.
    Supports generic OpenAI-compatible models if base_url is provided.
    """
    normalized_name = normalize_model_name(name)

    if normalized_name.startswith("deepseek-") and "base_url" not in kwargs:
        if not (os.getenv("OPENAI_BASE_URL") or os.getenv("DEFENSE_BASE_URL")):
            kwargs["base_url"] = "https://api.deepseek.com"

    provider = _resolve_provider(normalized_name, kwargs)

    if not provider:
        raise ValueError(
            f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}.\n"
            "For a custom OpenAI-compatible model, provide 'base_url' or set "
            "provider='openai'."
        )

    try:
        backend_class = _load_backend_class(provider)
    except ModuleNotFoundError as exc:
        raise ValueError(_build_missing_dependency_message(normalized_name, exc)) from exc

    if "provider" in kwargs:
        del kwargs["provider"]

    init_kwargs, dropped_kwargs = _filter_backend_kwargs(backend_class, kwargs)
    if dropped_kwargs:
        warnings.warn(
            f"{backend_class.__name__} ignored unsupported kwargs: {dropped_kwargs}",
            stacklevel=2,
        )
    return backend_class(model=normalized_name, **init_kwargs)


def list_available_backends() -> list[str]:
    """List all configured backend model names without importing SDKs."""
    return list(MODEL_REGISTRY.keys())


def get_backend_info(name: str) -> dict[str, Any]:
    """Get backend provider info without initializing backend clients."""
    normalized = normalize_model_name(name)
    provider = _resolve_provider(normalized) or "unknown"
    return {"provider": provider, "model": normalized}
