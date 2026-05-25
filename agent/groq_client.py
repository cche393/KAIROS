"""Provider-aware hosted LLM client helpers for KAIROS."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"


def load_environment() -> None:
    """Load local .env values when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def get_llm_config(load_env_file: bool = True) -> dict[str, Any]:
    """Return provider/model/API-key status without exposing the key value."""
    if load_env_file:
        load_environment()
    provider = os.getenv("KAIROS_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    model = os.getenv("KAIROS_LLM_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    warnings = []
    if provider == "groq" and _looks_like_openai_model(model):
        warnings.append(
            f"KAIROS_LLM_MODEL={model} is not a Groq model; using {DEFAULT_GROQ_MODEL}."
        )
        model = DEFAULT_GROQ_MODEL
    key_name = _api_key_name(provider)
    return {
        "provider": provider,
        "model": model,
        "api_key_name": key_name,
        "api_key_configured": bool(os.getenv(key_name, "").strip()),
        "warnings": warnings,
    }


def request_chat_completion(
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None = None,
    temperature: float = 0,
    load_env_file: bool = True,
) -> str:
    """Call the configured hosted chat-completion provider and return text."""
    config = get_llm_config(load_env_file=load_env_file)
    provider = config["provider"]
    if provider == "groq":
        return _request_groq(messages, config["model"], response_format, temperature)
    if provider == "openai":
        return _request_openai(messages, config["model"], response_format, temperature)
    raise RuntimeError(f"Unsupported KAIROS_LLM_PROVIDER: {provider}")


def _request_groq(
    messages: list[dict[str, str]],
    model: str,
    response_format: dict[str, Any] | None,
    temperature: float,
) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    try:
        from groq import Groq
    except Exception as exc:
        raise RuntimeError(f"Groq SDK import failed: {exc}") from exc

    client = Groq(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or "{}"


def _request_openai(
    messages: list[dict[str, str]],
    model: str,
    response_format: dict[str, Any] | None,
    temperature: float,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"OpenAI SDK import failed: {exc}") from exc

    client = OpenAI(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or "{}"


def _api_key_name(provider: str) -> str:
    if provider == "openai":
        return "OPENAI_API_KEY"
    return "GROQ_API_KEY"


def _looks_like_openai_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith(("gpt-", "o1", "o3", "o4"))
