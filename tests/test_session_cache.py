"""Tests for the long-lived session hijacking / prompt caching module."""

from __future__ import annotations

import pytest

from token_tamer.session_cache import (
    KEEP_TRAILING_MESSAGES,
    MAX_BREAKPOINTS,
    MIN_MESSAGES_TO_CACHE,
    SessionCache,
)


def _msg(role: str, content) -> dict:
    return {"role": role, "content": content}


@pytest.fixture
def cache():
    return SessionCache()


# ──────────────────────────────────────────────────────────
#  Tools array tagging
# ──────────────────────────────────────────────────────────


class TestToolsCaching:
    def test_tools_array_gets_cache_control(self, cache):
        body = {
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hello"),
                _msg("user", "do something"),
            ],
            "tools": [
                {"name": "Read", "input_schema": {"type": "object"}},
                {"name": "Edit", "input_schema": {"type": "object"}},
            ],
        }
        modified, info = cache.apply(body)
        # Last tool tagged
        assert modified["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        # First tool untouched
        assert "cache_control" not in modified["tools"][0]
        assert info["breakpoints"] >= 1

    def test_empty_tools_array_not_tagged(self, cache):
        body = {
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hello"),
                _msg("user", "do something"),
            ],
            "tools": [],
        }
        modified, info = cache.apply(body)
        assert modified["tools"] == []

    def test_no_tools_field_is_fine(self, cache):
        body = {
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hello"),
                _msg("user", "do something"),
            ],
        }
        modified, info = cache.apply(body)
        assert "tools" not in modified

    def test_tools_double_apply_is_idempotent(self, cache):
        body = {
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hi"),
                _msg("user", "go"),
            ],
            "tools": [{"name": "Read"}],
        }
        cache.apply(body)
        breakpoints_after_first = sum(
            1 for t in body["tools"] if "cache_control" in t
        )
        cache.apply(body)
        breakpoints_after_second = sum(
            1 for t in body["tools"] if "cache_control" in t
        )
        assert breakpoints_after_first == breakpoints_after_second == 1


# ──────────────────────────────────────────────────────────
#  System prompt tagging
# ──────────────────────────────────────────────────────────


class TestSystemCaching:
    def test_string_system_promoted_to_list(self, cache):
        body = {
            "system": "You are Claude Code, a helpful assistant.",
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hi"),
                _msg("user", "go"),
            ],
        }
        modified, info = cache.apply(body)
        # String got normalized to list with cache_control
        assert isinstance(modified["system"], list)
        assert modified["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert modified["system"][0]["text"] == "You are Claude Code, a helpful assistant."

    def test_list_system_last_block_tagged(self, cache):
        body = {
            "system": [
                {"type": "text", "text": "block 1"},
                {"type": "text", "text": "block 2"},
            ],
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hi"),
                _msg("user", "go"),
            ],
        }
        modified, info = cache.apply(body)
        assert "cache_control" not in modified["system"][0]
        assert modified["system"][1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_system_not_tagged(self, cache):
        body = {
            "system": "",
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hi"),
                _msg("user", "go"),
            ],
        }
        modified, info = cache.apply(body)
        assert modified["system"] == ""


# ──────────────────────────────────────────────────────────
#  Conversation prefix tagging
# ──────────────────────────────────────────────────────────


class TestConversationPrefix:
    def test_short_convo_not_tagged(self, cache):
        body = {
            "messages": [
                _msg("user", "hi"),
                _msg("assistant", "hello"),
            ],
        }
        modified, info = cache.apply(body)
        assert info["breakpoints"] == 0

    def test_long_convo_string_content_promoted(self, cache):
        msgs = [
            _msg("user", f"turn {i} content") for i in range(5)
        ]
        body = {"messages": msgs}
        modified, info = cache.apply(body)
        # Cutoff is len-2 = 3, target is index 2
        target = modified["messages"][len(msgs) - KEEP_TRAILING_MESSAGES - 1]
        assert isinstance(target["content"], list)
        assert target["content"][0]["cache_control"] == {"type": "ephemeral"}
        # The trailing 2 messages stay untouched (still strings or untagged)
        for trailing in modified["messages"][-KEEP_TRAILING_MESSAGES:]:
            content = trailing["content"]
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        assert "cache_control" not in blk

    def test_long_convo_list_content_tagged(self, cache):
        msgs = [
            _msg("user", [{"type": "text", "text": f"hi {i}"}])
            for i in range(5)
        ]
        body = {"messages": msgs}
        modified, info = cache.apply(body)
        target_idx = len(msgs) - KEEP_TRAILING_MESSAGES - 1
        last_block = modified["messages"][target_idx]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}

    def test_tool_use_block_can_be_tagged(self, cache):
        msgs = [
            _msg("user", "fix payment.py"),
            _msg("assistant", [
                {"type": "text", "text": "Reading"},
                {"type": "tool_use", "id": "t1", "name": "Read",
                 "input": {"path": "payment.py"}},
            ]),
            _msg("user", [{"type": "tool_result", "tool_use_id": "t1",
                            "content": "file body"}]),
            _msg("assistant", "Looking..."),
            _msg("user", "what's next?"),
        ]
        body = {"messages": msgs}
        modified, info = cache.apply(body)
        target_idx = len(msgs) - KEEP_TRAILING_MESSAGES - 1
        target = modified["messages"][target_idx]
        # Some block in the target message should have cache_control
        tagged_blocks = [
            blk for blk in target["content"]
            if isinstance(blk, dict) and "cache_control" in blk
        ]
        assert len(tagged_blocks) == 1


# ──────────────────────────────────────────────────────────
#  Limits and idempotency
# ──────────────────────────────────────────────────────────


class TestLimitsAndIdempotency:
    def test_total_breakpoints_within_budget(self, cache):
        msgs = [_msg("user", f"turn {i}") for i in range(8)]
        body = {
            "system": "System prompt",
            "messages": msgs,
            "tools": [{"name": "Read"}],
        }
        modified, info = cache.apply(body)
        assert info["breakpoints"] <= MAX_BREAKPOINTS

    def test_double_apply_does_not_add_extra_breakpoints(self, cache):
        msgs = [_msg("user", f"turn {i}") for i in range(8)]
        body = {
            "system": "System prompt",
            "messages": msgs,
            "tools": [{"name": "Read"}],
        }
        _, info1 = cache.apply(body)
        _, info2 = cache.apply(body)
        # The second apply should add 0 new breakpoints (all already there)
        # We verify by counting cache_control markers in the modified body
        markers = _count_cache_markers(body)
        assert markers == info1["breakpoints"]


def _count_cache_markers(body: dict) -> int:
    count = 0
    sys_field = body.get("system")
    if isinstance(sys_field, list):
        for blk in sys_field:
            if isinstance(blk, dict) and "cache_control" in blk:
                count += 1
    for t in body.get("tools") or []:
        if isinstance(t, dict) and "cache_control" in t:
            count += 1
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and "cache_control" in blk:
                    count += 1
    return count


# ──────────────────────────────────────────────────────────
#  Session tracking
# ──────────────────────────────────────────────────────────


class TestSessionTracking:
    def test_same_prefix_gets_same_session_id(self, cache):
        msgs = [
            _msg("user", "Let's fix payment.py"),
            _msg("assistant", "Sure, looking now"),
        ]
        body1 = {"messages": msgs + [_msg("user", "turn 3")]}
        body2 = {"messages": msgs + [_msg("user", "turn 4")]}
        _, info1 = cache.apply(body1)
        _, info2 = cache.apply(body2)
        assert info1["session_id"] == info2["session_id"]
        assert info2["turn_count"] == 2  # Same session, second turn

    def test_different_prefixes_get_different_session_ids(self, cache):
        body1 = {"messages": [
            _msg("user", "Task A"),
            _msg("assistant", "ok"),
            _msg("user", "more"),
        ]}
        body2 = {"messages": [
            _msg("user", "Task B"),
            _msg("assistant", "ok"),
            _msg("user", "more"),
        ]}
        _, info1 = cache.apply(body1)
        _, info2 = cache.apply(body2)
        assert info1["session_id"] != info2["session_id"]

    def test_stats_counter_increments(self, cache):
        # Two distinct conversations (real-world: each turn is a fresh body)
        cache.apply({"messages": [_msg("user", f"task A {i}") for i in range(4)]})
        cache.apply({"messages": [_msg("user", f"task B {i}") for i in range(4)]})
        assert cache.stats.requests_with_breakpoints == 2
        assert cache.stats.breakpoints_injected >= 2
        assert cache.stats.sessions_seen == 2


# ──────────────────────────────────────────────────────────
#  Token estimation
# ──────────────────────────────────────────────────────────


class TestTokenEstimation:
    def test_estimate_grows_with_prefix(self, cache):
        small = {"messages": [
            _msg("user", "a"),
            _msg("assistant", "b"),
            _msg("user", "c"),
        ]}
        large = {"messages": [
            _msg("user", "x" * 4000),
            _msg("assistant", "y" * 4000),
            _msg("user", "now do this"),
        ]}
        _, info_small = cache.apply(small)
        _, info_large = cache.apply(large)
        assert info_large["cached_tokens_estimate"] > info_small["cached_tokens_estimate"]
        # Roughly 1000 tokens for each 4000-char block, but only the prefix
        # (one block) is cached when KEEP_TRAILING_MESSAGES=2 → so ~1000
        assert info_large["cached_tokens_estimate"] >= 900


# ──────────────────────────────────────────────────────────
#  Robustness
# ──────────────────────────────────────────────────────────


class TestRobustness:
    def test_empty_body(self, cache):
        body, info = cache.apply({})
        assert info["breakpoints"] == 0

    def test_messages_field_missing(self, cache):
        body, info = cache.apply({"system": "hi"})
        assert info["breakpoints"] == 0

    def test_garbage_messages_dont_crash(self, cache):
        body = {"messages": [None, "string", 42, {"role": "user"}]}
        # Should not raise
        result, info = cache.apply(body)
        assert info is not None
