"""
Token counting and cost estimation using tiktoken.

Provides accurate BPE-based token counting for OpenAI and Anthropic models,
plus cumulative session metrics tracking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tiktoken


@dataclass
class FileStats:
    """Token statistics for a single file."""
    filename: str
    original_tokens: int = 0
    compressed_tokens: int = 0
    was_skeletonized: bool = False

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: float = 0.0
    original_tokens: int = 0
    compressed_tokens: int = 0
    file_stats: List[FileStats] = field(default_factory=list)
    model: str = ""

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)


@dataclass
class SessionMetrics:
    """Cumulative session-wide metrics."""
    start_time: float = field(default_factory=time.time)
    total_requests: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    cost_saved: float = 0.0
    recent_requests: List[RequestMetrics] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def record_request(self, metrics: RequestMetrics, cost_saved: float = 0.0) -> None:
        """Thread-safe recording of a request's metrics."""
        with self._lock:
            self.total_requests += 1
            self.original_tokens += metrics.original_tokens
            self.compressed_tokens += metrics.compressed_tokens
            self.cost_saved += cost_saved
            self.recent_requests.append(metrics)
            # Keep only last 50 requests in memory
            if len(self.recent_requests) > 50:
                self.recent_requests = self.recent_requests[-50:]

    @property
    def latest_request(self) -> Optional[RequestMetrics]:
        """Get the most recent request metrics."""
        if self.recent_requests:
            return self.recent_requests[-1]
        return None


class TokenCounter:
    """
    Counts tokens using tiktoken and estimates costs based on model pricing.
    """

    # Cache for encodings to avoid repeated loading
    _encoding_cache: Dict[str, tiktoken.Encoding] = {}

    def __init__(self) -> None:
        # Pre-load the most common encoding
        self._default_encoding = tiktoken.get_encoding("cl100k_base")

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get the tiktoken encoding for a model, with caching."""
        if model in self._encoding_cache:
            return self._encoding_cache[model]

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for unknown models (covers most modern models)
            encoding = self._default_encoding

        self._encoding_cache[model] = encoding
        return encoding

    def count(self, text: str, model: str = "gpt-4o") -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.
            model: The model name to use for encoding selection.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        encoding = self._get_encoding(model)
        return len(encoding.encode(text))

    def count_messages(self, messages: List[dict], model: str = "gpt-4o") -> int:
        """
        Count tokens in a messages array (OpenAI chat format).

        Includes overhead tokens for message formatting.
        """
        total = 0
        for message in messages:
            # Every message has ~4 tokens of overhead (role, separators)
            total += 4
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count(content, model)
            elif isinstance(content, list):
                # Multi-part content (e.g., with images)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += self.count(part.get("text", ""), model)
        # Every reply is primed with <|start|>assistant<|message|>
        total += 3
        return total

    @staticmethod
    def estimate_cost(
        input_tokens: int,
        output_tokens: int,
        input_price_per_million: float,
        output_price_per_million: float,
    ) -> float:
        """
        Estimate cost in USD for a given number of input/output tokens.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            input_price_per_million: Price per 1M input tokens in USD.
            output_price_per_million: Price per 1M output tokens in USD.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (input_tokens / 1_000_000) * input_price_per_million
        output_cost = (output_tokens / 1_000_000) * output_price_per_million
        return input_cost + output_cost
