"""
Configuration loader for TokenTamer.

Loads settings from config.yaml, merging with environment variable overrides.
Priority: environment variables > config file values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class UpstreamConfig:
    openai_url: str = "https://api.openai.com"
    anthropic_url: str = "https://api.anthropic.com"
    # ChatGPT-subscription backend used by Codex CLI when the user is signed in
    # with a ChatGPT Plus/Pro/Team account instead of an OpenAI API key.
    chatgpt_backend_url: str = "https://chatgpt.com/backend-api/codex"


@dataclass
class ApiKeysConfig:
    openai: str = ""
    anthropic: str = ""


@dataclass
class SkeletonizerConfig:
    keep_docstrings: bool = False
    keep_class_attrs: bool = True


@dataclass
class ContextConfig:
    repo_path: str = ""


@dataclass
class ModelPricing:
    """Pricing per 1M tokens."""
    input: float = 3.00
    output: float = 15.00


@dataclass
class Config:
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    api_keys: ApiKeysConfig = field(default_factory=ApiKeysConfig)
    skeletonizer: SkeletonizerConfig = field(default_factory=SkeletonizerConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    pricing: Dict[str, ModelPricing] = field(default_factory=dict)

    @property
    def repo_path(self) -> Optional[str]:
        path = self.context.repo_path
        if path and Path(path).exists():
            return str(Path(path).resolve())
        return None

    def get_api_key(self, provider: str, request_key: Optional[str] = None) -> str:
        """
        Resolve API key with priority: request header > env var > config file.
        """
        if request_key:
            return request_key

        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY", self.api_keys.openai)
        elif provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", self.api_keys.anthropic)
        return ""

    def get_pricing(self, model: str) -> ModelPricing:
        """Get pricing for a model, falling back to default."""
        if model in self.pricing:
            return self.pricing[model]
        return self.pricing.get("default", ModelPricing())


def load_config(config_path: Union[str, Path, None] = None) -> Config:
    """
    Load configuration from a YAML file, with environment variable overrides.
    """
    raw: dict[str, Any] = {}

    # Try loading from file
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
    else:
        # Search common locations
        for candidate in [
            Path("config.yaml"),
            Path("token_guard.yaml"),
            Path.home() / ".config" / "token-tamer" / "config.yaml",
        ]:
            if candidate.exists():
                with open(candidate, "r") as f:
                    raw = yaml.safe_load(f) or {}
                break

    # Build config from raw dict
    config = Config()

    # Proxy settings
    if "proxy" in raw:
        config.proxy = ProxyConfig(
            host=raw["proxy"].get("host", config.proxy.host),
            port=int(raw["proxy"].get("port", config.proxy.port)),
        )

    # Upstream URLs
    if "upstream" in raw:
        config.upstream = UpstreamConfig(
            openai_url=raw["upstream"].get("openai_url", config.upstream.openai_url),
            anthropic_url=raw["upstream"].get("anthropic_url", config.upstream.anthropic_url),
            chatgpt_backend_url=raw["upstream"].get(
                "chatgpt_backend_url", config.upstream.chatgpt_backend_url
            ),
        )

    # API keys (env vars take priority)
    if "api_keys" in raw:
        config.api_keys = ApiKeysConfig(
            openai=os.environ.get("OPENAI_API_KEY", raw["api_keys"].get("openai", "")),
            anthropic=os.environ.get("ANTHROPIC_API_KEY", raw["api_keys"].get("anthropic", "")),
        )
    else:
        config.api_keys = ApiKeysConfig(
            openai=os.environ.get("OPENAI_API_KEY", ""),
            anthropic=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    # Context settings
    if "context" in raw:
        config.context = ContextConfig(
            repo_path=raw["context"].get("repo_path", ""),
        )

    # Skeletonizer settings
    if "skeletonizer" in raw:
        config.skeletonizer = SkeletonizerConfig(
            keep_docstrings=raw["skeletonizer"].get("keep_docstrings", False),
            keep_class_attrs=raw["skeletonizer"].get("keep_class_attrs", True),
        )

    # Pricing
    if "pricing" in raw:
        for model_name, prices in raw["pricing"].items():
            config.pricing[model_name] = ModelPricing(
                input=float(prices.get("input", 3.00)),
                output=float(prices.get("output", 15.00)),
            )
    else:
        config.pricing["default"] = ModelPricing()

    return config
