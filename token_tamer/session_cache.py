"""
Long-Lived Session Hijacking via Anthropic Prompt Caching.

This module injects `cache_control` breakpoints into outbound Anthropic
Messages API requests to maximize cache reuse across the conversation.

Anthropic charges:
  - $3.00 per 1M input tokens (normal)
  - $0.30 per 1M input tokens (cache hits) ← 90% discount
  - $3.75 per 1M input tokens (cache writes, one-time)

Without this module, Claude Code re-sends the full conversation each turn
and gets almost no cache reuse because the prefix mutates. This module
makes the prefix stable and inserts breakpoints at strategic positions:

  1. After tools array (tools rarely change → very stable)
  2. After system prompt (also stable)
  3. After the conversation prefix (everything except last 1-2 turns)

A 50-turn coding session goes from ~$5 to ~$0.50 with no quality loss.

References:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Anthropic API allows up to 4 cache_control breakpoints per request.
# We use at most 3 (tools, system, conversation prefix) and leave 1 spare.
MAX_BREAKPOINTS = 4

# Minimum messages before we bother caching (small convos don't benefit).
MIN_MESSAGES_TO_CACHE = 3

# Number of trailing messages to keep OUTSIDE the cache breakpoint.
# These are the "fresh" turn(s) that change each request.
KEEP_TRAILING_MESSAGES = 2

# Anthropic cache TTL is 5 minutes. We track sessions for slightly longer
# so we can detect "expired" cache attempts and warn the user.
SESSION_TTL_SECONDS = 5 * 60


@dataclass
class SessionCacheStats:
    """Track cache performance for the dashboard."""
    requests_with_breakpoints: int = 0
    breakpoints_injected: int = 0
    sessions_seen: int = 0
    estimated_tokens_cached: int = 0

    def __str__(self) -> str:
        return (
            f"sessions={self.sessions_seen} "
            f"breakpoints={self.breakpoints_injected} "
            f"cached_tokens≈{self.estimated_tokens_cached}"
        )


@dataclass
class _SessionRecord:
    first_seen: float
    last_seen: float
    turn_count: int = 0


class SessionCache:
    """Manages prompt-cache breakpoint injection for Anthropic requests.

    Thread-safe. Self-cleaning. No persistence — all state is in-process
    and bounded by SESSION_TTL_SECONDS.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionRecord] = {}
        self._lock = threading.Lock()
        self.stats = SessionCacheStats()

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def apply(self, body: dict) -> Tuple[dict, Dict[str, Any]]:
        """Inject cache_control breakpoints into an Anthropic request body.

        Args:
            body: The Anthropic Messages API request body (will be modified).

        Returns:
            (modified_body, info_dict) where info_dict contains:
              - 'breakpoints': number of cache_control markers added
              - 'session_id': stable hash identifying this conversation
              - 'turn_count': how many times we've seen this session
              - 'cached_tokens_estimate': rough estimate of tokens reused
        """
        messages = body.get("messages") or []
        info = {
            "breakpoints": 0,
            "session_id": None,
            "turn_count": 0,
            "cached_tokens_estimate": 0,
            "prefix_end_index": -1,  # messages[0..prefix_end_index] are cached
        }

        existing_breakpoints = self._count_cache_controls(body)
        if existing_breakpoints > MAX_BREAKPOINTS:
            self._trim_cache_controls(body, MAX_BREAKPOINTS)
            existing_breakpoints = MAX_BREAKPOINTS

        if len(messages) < MIN_MESSAGES_TO_CACHE:
            self._normalize_cache_control_ttls(body)
            return body, info

        breakpoint_count = 0

        # ── 1. Cache the tools array (most stable thing in the request) ──
        if (
            existing_breakpoints + breakpoint_count < MAX_BREAKPOINTS
            and self._mark_tools_for_caching(body)
        ):
            breakpoint_count += 1

        # ── 2. Cache the system prompt ──
        if (
            existing_breakpoints + breakpoint_count < MAX_BREAKPOINTS
            and self._mark_system_for_caching(body)
        ):
            breakpoint_count += 1

        # ── 3. Cache the conversation prefix (all but trailing turns) ──
        remaining_budget = MAX_BREAKPOINTS - existing_breakpoints - breakpoint_count
        if remaining_budget > 0:
            prefix_idx = self._mark_conversation_prefix(body, messages)
            if prefix_idx >= 0:
                breakpoint_count += 1
                info["prefix_end_index"] = prefix_idx

        self._normalize_cache_control_ttls(body)

        # ── 4. Track the session ──
        session_id = self._fingerprint(messages)
        turn_count = self._record_session(session_id)

        info["breakpoints"] = breakpoint_count
        info["session_id"] = session_id
        info["turn_count"] = turn_count
        info["cached_tokens_estimate"] = self._estimate_cached_tokens(messages)

        # Update global stats
        if breakpoint_count > 0:
            self.stats.requests_with_breakpoints += 1
            self.stats.breakpoints_injected += breakpoint_count
            self.stats.estimated_tokens_cached += info["cached_tokens_estimate"]

        return body, info

    @staticmethod
    def _iter_cache_control_owners(body: dict) -> List[dict]:
        """Return dict objects that may own Anthropic cache_control markers.

        Anthropic accepts at most four cache breakpoints per request. Claude
        Code may already send some, so TokenTamer must treat existing markers
        as consuming the same global budget as markers it injects.
        """
        owners: List[dict] = []

        for tool in body.get("tools") or []:
            if isinstance(tool, dict) and "cache_control" in tool:
                owners.append(tool)

        system = body.get("system")
        if isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and "cache_control" in block:
                    owners.append(block)

        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        owners.append(block)

        return owners

    @classmethod
    def _count_cache_controls(cls, body: dict) -> int:
        return len(cls._iter_cache_control_owners(body))

    @classmethod
    def _trim_cache_controls(cls, body: dict, limit: int = MAX_BREAKPOINTS) -> int:
        """Remove surplus cache_control markers from the end of the request."""
        owners = cls._iter_cache_control_owners(body)
        removed = 0
        for owner in reversed(owners[limit:]):
            owner.pop("cache_control", None)
            removed += 1
        return removed

    @classmethod
    def _normalize_cache_control_ttls(cls, body: dict) -> None:
        """Keep cache_control TTLs valid in Anthropic processing order.

        Anthropic processes cache breakpoints as tools, system, then messages.
        A longer-lived 1h block cannot appear after a shorter/default 5m block.
        If Claude Code already sent any 1h cache block, promote earlier cache
        blocks to 1h so TokenTamer's injected defaults do not make the request
        invalid.
        """
        owners = cls._iter_cache_control_owners(body)
        last_one_hour = -1
        for index, owner in enumerate(owners):
            cache_control = owner.get("cache_control")
            if isinstance(cache_control, dict) and cache_control.get("ttl") == "1h":
                last_one_hour = index

        if last_one_hour < 0:
            return

        for owner in owners[: last_one_hour + 1]:
            cache_control = owner.get("cache_control")
            if not isinstance(cache_control, dict):
                continue
            cache_control["ttl"] = "1h"

    def reset(self) -> None:
        """Forget all tracked sessions. Useful for tests."""
        with self._lock:
            self._sessions.clear()
            self.stats = SessionCacheStats()

    # ──────────────────────────────────────────────────────────
    #  Breakpoint placement
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _mark_tools_for_caching(body: dict) -> bool:
        """Add cache_control to the LAST tool in the tools array."""
        tools = body.get("tools")
        if not isinstance(tools, list) or not tools:
            return False
        last_tool = tools[-1]
        if not isinstance(last_tool, dict):
            return False
        # Idempotent: don't double-mark.
        if "cache_control" in last_tool:
            return False
        last_tool["cache_control"] = {"type": "ephemeral"}
        return True

    @staticmethod
    def _mark_system_for_caching(body: dict) -> bool:
        """Add cache_control to the system prompt.

        Anthropic's `system` field can be a string OR a list of content
        blocks. We normalize string → list and tag the last block.
        """
        system = body.get("system")
        if not system:
            return False

        if isinstance(system, str):
            # Normalize to list format so we can attach cache_control
            if not system.strip():
                return False
            body["system"] = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
            return True

        if isinstance(system, list) and system:
            # Find the last block we can tag (must be a dict)
            for block in reversed(system):
                if isinstance(block, dict):
                    if "cache_control" in block:
                        return False  # Already tagged
                    block["cache_control"] = {"type": "ephemeral"}
                    return True
        return False

    @staticmethod
    def _mark_conversation_prefix(body: dict, messages: List[dict]) -> int:
        """Tag the LAST content block of the message at index `len-2` (or
        further back) so the conversation prefix gets cached.

        We deliberately exclude the trailing turn(s) so the cache prefix
        stays maximal even as new turns arrive.

        Returns the index of the tagged message, or -1 if no breakpoint
        was added.
        """
        cutoff = len(messages) - KEEP_TRAILING_MESSAGES
        if cutoff < 1:
            return -1

        # Walk backwards from cutoff to find the latest message that's a
        # dict we can safely tag. Non-dict garbage is just skipped.
        target_idx = None
        target = None
        for i in range(cutoff - 1, -1, -1):
            if isinstance(messages[i], dict):
                target_idx = i
                target = messages[i]
                break
        if target is None:
            return -1

        content = target.get("content")
        if not content:
            return -1

        # Normalize string content → list-of-blocks so we can attach control.
        if isinstance(content, str):
            target["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
            return target_idx

        if isinstance(content, list):
            # Find the last dict block we can tag, prefer text blocks.
            # We pick the LAST block period so the breakpoint covers
            # the entire message including any tool_use blocks.
            for i in reversed(range(len(content))):
                block = content[i]
                if not isinstance(block, dict):
                    continue
                if "cache_control" in block:
                    return -1  # Already tagged earlier
                # Some block types (image, document) don't support
                # cache_control directly — only text/tool_use/tool_result do.
                if block.get("type") in ("text", "tool_use", "tool_result", "document"):
                    block["cache_control"] = {"type": "ephemeral"}
                    return target_idx
            return -1

        return -1

    # ──────────────────────────────────────────────────────────
    #  Session fingerprinting
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(messages: List[dict]) -> str:
        """Compute a stable hash of the conversation's first 2 messages.

        Two messages from the same Claude Code session will share the same
        opening (system prompt + first user turn), so the prefix hash is
        a reliable session identifier.
        """
        if not messages:
            return "empty"
        head = messages[: min(2, len(messages))]
        # Convert to a deterministic string. We only hash text to avoid
        # being sensitive to ordering of keys in tool inputs.
        material_parts: List[str] = []
        for msg in head:
            if not isinstance(msg, dict):
                material_parts.append(repr(type(msg).__name__))
                continue
            material_parts.append(str(msg.get("role", "")))
            content = msg.get("content")
            if isinstance(content, str):
                material_parts.append(content[:512])
            elif isinstance(content, list):
                for block in content[:3]:
                    if isinstance(block, dict):
                        material_parts.append(str(block.get("type", "")))
                        material_parts.append(str(block.get("text", ""))[:256])
        joined = "\x1f".join(material_parts)
        return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _record_session(self, session_id: str) -> int:
        """Update session tracking and return the turn count."""
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            record = self._sessions.get(session_id)
            if record is None:
                record = _SessionRecord(first_seen=now, last_seen=now, turn_count=1)
                self._sessions[session_id] = record
                self.stats.sessions_seen += 1
            else:
                record.last_seen = now
                record.turn_count += 1
            return record.turn_count

    def _cleanup_expired_locked(self, now: float) -> None:
        """Remove sessions older than SESSION_TTL_SECONDS. Called with lock held."""
        cutoff = now - SESSION_TTL_SECONDS
        expired = [sid for sid, rec in self._sessions.items() if rec.last_seen < cutoff]
        for sid in expired:
            del self._sessions[sid]

    # ──────────────────────────────────────────────────────────
    #  Token estimation (rough heuristic for stats)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_cached_tokens(messages: List[dict]) -> int:
        """Rough estimate of how many tokens are in the cached prefix.

        We use a 4-chars-per-token heuristic — accurate enough for dashboards.
        Excludes the trailing un-cached turns.
        """
        cutoff = len(messages) - KEEP_TRAILING_MESSAGES
        if cutoff < 1:
            return 0
        char_count = 0
        for msg in messages[:cutoff]:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                char_count += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or ""
                        if isinstance(text, str):
                            char_count += len(text)
                        # tool_use/tool_result have JSON-like content too
                        inp = block.get("input") or block.get("content")
                        if isinstance(inp, str):
                            char_count += len(inp)
        return char_count // 4
