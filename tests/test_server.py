"""Tests for the FastAPI proxy server."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from token_tamer.config import Config
from token_tamer.server import create_app
from token_tamer.token_counter import SessionMetrics


@pytest.fixture
def config():
    cfg = Config()
    cfg.upstream.openai_url = "https://fake-openai.com"
    cfg.upstream.anthropic_url = "https://fake-anthropic.com"
    return cfg


@pytest.fixture
def metrics():
    return SessionMetrics()


@pytest.fixture
def app(config, metrics):
    return create_app(config, metrics)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealth:
    def test_health_endpoint(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        from token_tamer import __version__
        assert data["version"] == __version__
        assert data["requests_processed"] == 0


class TestOpenAIProxy:
    @patch("httpx.AsyncClient.post")
    def test_chat_completions_compression(self, mock_post, client: TestClient):
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "chatcmpl-test", "choices": []}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Fix payment.py\n\n"
                        "```python\n# File: payment.py\ndef calc():\n    return 1+1\n```\n\n"
                        "```python\n# File: utils.py\ndef helper():\n    x=1\n    y=2\n    return x+y\n```"
                    ),
                }
            ],
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.headers["X-TokenTamer-Saved"]
        saved = int(response.headers["X-TokenTamer-Saved"])
        assert saved > 0

        # Verify the forwarded body was compressed
        call_args = mock_post.call_args
        forwarded_body = call_args.kwargs["json"]
        assert "payment.py" in forwarded_body["messages"][0]["content"]
        assert "..." in forwarded_body["messages"][0]["content"]

    @patch("httpx.AsyncClient.post")
    def test_chat_completions_streaming(self, mock_post, client: TestClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_post.return_value = mock_response

        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    @patch("httpx.AsyncClient.request")
    def test_models_pass_through(self, mock_request, client: TestClient):
        mock_response = MagicMock()
        mock_response.content = b'{"data": []}'
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_request.return_value = mock_response

        response = client.get("/v1/models", headers={"Authorization": "Bearer test"})
        assert response.status_code == 200


class TestAnthropicProxy:
    @patch("httpx.AsyncClient.post")
    def test_messages_compression_with_system(self, mock_post, client: TestClient):
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "msg-test", "content": []}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "claude-sonnet-4-20250514",
            "system": "Focus on payment.py",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "```python\n# File: payment.py\ndef calc():\n    return 1+1\n```\n\n"
                        "```python\n# File: database.py\ndef connect():\n    pass\n```"
                    ),
                }
            ],
            "stream": False,
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        assert response.headers["X-TokenTamer-Saved"]

        # system prompt should have made payment.py active
        call_args = mock_post.call_args
        forwarded_body = call_args.kwargs["json"]
        content = forwarded_body["messages"][0]["content"]
        assert "..." in content  # database.py should be skeletonized


class TestMetrics:
    def test_metrics_tracked(self, client: TestClient, metrics: SessionMetrics):
        assert metrics.total_requests == 0

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.content = json.dumps({"id": "test"}).encode()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_post.return_value = mock_response

            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "```python\n"
                                "# File: foo.py\n"
                                "def calculate_tax(amount, region):\n"
                                "    rate = get_base_rate(region)\n"
                                "    adjustments = fetch_adjustments(region, amount)\n"
                                "    if amount > 10000:\n"
                                "        rate *= 1.05\n"
                                "    subtotal = amount * rate\n"
                                "    for adj in adjustments:\n"
                                "        subtotal += adj.value\n"
                                "    return round(subtotal, 2)\n"
                                "```"
                            ),
                        }
                    ],
                },
            )

        assert metrics.total_requests == 1
        assert metrics.tokens_saved >= 0


# ──────────────────────────────────────────────────────────
#  Safety guard tests: tool/function-call requests
# ──────────────────────────────────────────────────────────


def _heavy_code_block() -> str:
    return (
        "```python\n# File: utils.py\n"
        "def helper():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    z = 3\n"
        "    return x + y + z\n"
        "```"
    )


class TestToolSafety:
    @patch("httpx.AsyncClient.post")
    def test_openai_tools_request_uses_smart_compression(
        self, mock_post, client: TestClient
    ):
        """With tools present, plain-text code blocks should still be compressed."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "x", "choices": []}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        original_msg = f"Refactor it\n\n{_heavy_code_block()}"
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": original_msg}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "edit_file", "parameters": {}},
                }
            ],
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        # Tools present BUT it's a plain user message → smart compression skeletonizes
        assert "..." in forwarded["messages"][0]["content"]

    @patch("httpx.AsyncClient.post")
    def test_anthropic_tool_use_part_is_preserved(self, mock_post, client: TestClient):
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m", "content": []}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Looking at code\n{_heavy_code_block()}"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "str_replace",
                            "input": {"path": "utils.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1",
                         "content": "ok"},
                    ],
                },
            ],
            "tools": [{"name": "str_replace", "input_schema": {"type": "object"}}],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        # tool_use & tool_result parts must be present untouched
        assistant_parts = forwarded["messages"][0]["content"]
        kinds = [p.get("type") for p in assistant_parts]
        assert "tool_use" in kinds
        tool_result = forwarded["messages"][1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "toolu_1"


class TestPassthroughMode:
    @patch("httpx.AsyncClient.post")
    def test_passthrough_skips_compression(
        self, mock_post, config, metrics
    ):
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "x"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        app = create_app(config, metrics, passthrough=True)
        client = TestClient(app)

        original_msg = f"Refactor it\n\n{_heavy_code_block()}"
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": original_msg}],
            },
        )

        forwarded = mock_post.call_args.kwargs["json"]
        assert forwarded["messages"][0]["content"] == original_msg


# ──────────────────────────────────────────────────────────
#  Responses API (Codex CLI)
# ──────────────────────────────────────────────────────────


class TestResponsesAPI:
    @patch("httpx.AsyncClient.post")
    def test_responses_string_input_is_compressed(self, mock_post, client: TestClient):
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "resp_1"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "gpt-4o",
            "input": (
                f"Fix payment.py\n\n"
                f"```python\n# File: payment.py\ndef calc():\n    return 1+1\n```\n\n"
                f"{_heavy_code_block()}"
            ),
        }

        response = client.post("/v1/responses", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        assert "..." in forwarded["input"]  # utils.py skeletonized
        assert "payment.py" in forwarded["input"]

    @patch("httpx.AsyncClient.post")
    def test_responses_list_input_preserves_tool_items(self, mock_post, client: TestClient):
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "resp_2"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": f"hi\n{_heavy_code_block()}"},
                {"type": "function_call", "name": "shell",
                 "arguments": "{\"cmd\":\"ls\"}", "call_id": "c1"},
                {"type": "function_call_output",
                 "call_id": "c1", "output": "file.py"},
            ],
        }

        response = client.post("/v1/responses", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        kinds = [
            item.get("type") if isinstance(item, dict) else None
            for item in forwarded["input"]
        ]
        # tool items preserved at correct positions
        assert "function_call" in kinds
        assert "function_call_output" in kinds

    @patch("httpx.AsyncClient.post")
    def test_responses_with_tools_still_compresses_text(
        self, mock_post, client: TestClient
    ):
        """Responses API with tools should still skeletonize plain code blocks."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "resp_3"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        original = f"Refactor\n{_heavy_code_block()}"
        payload = {
            "model": "gpt-4o",
            "input": original,
            "tools": [{"type": "function", "name": "shell"}],
        }

        response = client.post("/v1/responses", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        # With smart compression, plain code blocks ARE compressed even with tools
        assert "..." in forwarded["input"]


# ──────────────────────────────────────────────────────────
#  Phase 2: Tool-aware compression
# ──────────────────────────────────────────────────────────


def _file_dump(filename: str, body: str = None) -> str:
    body = body or (
        "def calculate_tax(amount, region):\n"
        "    rate = get_base_rate(region)\n"
        "    adjustments = fetch_adjustments(region, amount)\n"
        "    if amount > 10000:\n"
        "        rate *= 1.05\n"
        "    subtotal = amount * rate\n"
        "    for adj in adjustments:\n"
        "        subtotal += adj.value\n"
        "    return round(subtotal, 2)\n"
    )
    return body


class TestToolAwareCompression:
    @patch("httpx.AsyncClient.post")
    def test_stale_tool_result_is_skeletonized(self, mock_post, client: TestClient):
        """When a file is read twice, the older tool_result should be skeletonized."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        file_body = _file_dump("payment.py")
        payload = {
            "model": "claude-sonnet-4-20250514",
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Fix payment.py"},
                # First read
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"file_path": "payment.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": file_body},
                ]},
                # Some intermediate chatter
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me re-read the file"},
                ]},
                # Second read of SAME file
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t2", "name": "Read",
                     "input": {"file_path": "payment.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2",
                     "content": file_body},
                ]},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        msgs = forwarded["messages"]

        # With cache-first design, the conversation prefix (msg 0..3) stays
        # byte-identical across turns so Anthropic's exact-prefix cache hits.
        # Msg 2 (stale tool_result) is inside the prefix → preserved intact.
        # Msg 5 (latest tool_result) is in the tail → also preserved (latest).
        first_result_content = msgs[2]["content"][0]["content"]
        last_result_content = msgs[5]["content"][0]["content"]

        # Both are preserved: msg 2 for cache stability, msg 5 as latest read
        assert first_result_content == file_body
        assert last_result_content == file_body

    @patch("httpx.AsyncClient.post")
    def test_single_tool_result_is_preserved(self, mock_post, client: TestClient):
        """A file read only once should never be skeletonized."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        file_body = _file_dump("only_once.py")
        payload = {
            "model": "claude-sonnet-4-20250514",
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Look at only_once.py"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"file_path": "only_once.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": file_body},
                ]},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        # Single read → preserved
        assert forwarded["messages"][2]["content"][0]["content"] == file_body

    @patch("httpx.AsyncClient.post")
    def test_tool_use_blocks_are_never_touched(self, mock_post, client: TestClient):
        """tool_use blocks (the command itself) must always pass through intact."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "claude-sonnet-4-20250514",
            "tools": [{"name": "Edit", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "u1", "name": "Edit",
                     "input": {"file_path": "x.py", "old_str": "a", "new_str": "b"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "u1", "content": "ok"},
                ]},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        tool_use = forwarded["messages"][0]["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["input"]["old_str"] == "a"
        assert tool_use["input"]["new_str"] == "b"

    @patch("httpx.AsyncClient.post")
    def test_anthropic_list_format_tool_result(self, mock_post, client: TestClient):
        """tool_result with list-of-blocks content format should work too."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        file_body = _file_dump("foo.py")
        payload = {
            "model": "claude-sonnet-4-20250514",
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [
                # Read 1
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"path": "foo.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": file_body}]},
                ]},
                # Read 2
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t2", "name": "Read",
                     "input": {"path": "foo.py"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2",
                     "content": [{"type": "text", "text": file_body}]},
                ]},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]
        # With cache-first, msg 1 (stale read) is in the cached prefix →
        # preserved intact for cross-turn cache stability.
        first = forwarded["messages"][1]["content"][0]["content"][0]["text"]
        assert first == file_body
        # Last read's text block is intact (latest read)
        last = forwarded["messages"][3]["content"][0]["content"][0]["text"]
        assert last == file_body

    @patch("httpx.AsyncClient.post")
    def test_no_tool_compression_flag_disables_smart_path(
        self, mock_post, config, metrics
    ):
        """--no-tool-compression should fall back to leaving tool requests alone."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        app = create_app(config, metrics, compress_with_tools=False)
        client = TestClient(app)

        original_msg = f"Refactor\n{_heavy_code_block()}"
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": original_msg}],
                "tools": [{"type": "function", "function": {"name": "x"}}],
            },
        )
        forwarded = mock_post.call_args.kwargs["json"]
        # With smart compression disabled and tools present → no compression
        assert forwarded["messages"][0]["content"] == original_msg


# ──────────────────────────────────────────────────────────
#  Long-Lived Session Hijacking via prompt caching
# ──────────────────────────────────────────────────────────


class TestSessionCacheIntegration:
    @patch("httpx.AsyncClient.post")
    def test_anthropic_request_gets_cache_breakpoints(
        self, mock_post, client: TestClient
    ):
        """Multi-turn Anthropic request should have cache_control injected."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "claude-sonnet-4-20250514",
            "system": "You are a helpful coding assistant.",
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Fix payment.py"},
                {"role": "assistant", "content": "Looking now..."},
                {"role": "user", "content": "Anything else?"},
                {"role": "assistant", "content": "I see one issue."},
                {"role": "user", "content": "Apply the fix."},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200

        # Response headers expose breakpoint count
        assert int(response.headers["X-TokenTamer-Cache-Breakpoints"]) >= 2
        assert int(response.headers["X-TokenTamer-Cache-Tokens"]) > 0

        # The forwarded payload should contain cache_control markers
        forwarded = mock_post.call_args.kwargs["json"]

        # Tools array tagged
        assert forwarded["tools"][-1].get("cache_control") == {"type": "ephemeral"}

        # System prompt promoted to list with cache_control
        assert isinstance(forwarded["system"], list)
        assert any(
            isinstance(b, dict) and "cache_control" in b
            for b in forwarded["system"]
        )

        # Some message in the prefix has cache_control
        prefix_markers = 0
        for msg in forwarded["messages"][:-2]:  # exclude last 2 (trailing)
            content = msg.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and "cache_control" in blk:
                        prefix_markers += 1
        assert prefix_markers >= 1

    @patch("httpx.AsyncClient.post")
    def test_no_session_cache_flag_disables_breakpoints(
        self, mock_post, config, metrics
    ):
        """--no-session-cache should leave the body untouched."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        app = create_app(config, metrics, session_cache_enabled=False)
        client = TestClient(app)

        payload = {
            "model": "claude-sonnet-4-20250514",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "more"},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]

        # No cache_control anywhere
        assert forwarded["system"] == "You are helpful."  # not promoted to list
        for msg in forwarded["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        assert "cache_control" not in blk

    @patch("httpx.AsyncClient.post")
    def test_short_conversation_skips_caching(self, mock_post, client: TestClient):
        """Single-turn conversations shouldn't get caching (no benefit)."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "what's 2+2?"},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        # 1 message < MIN_MESSAGES_TO_CACHE → no conversation breakpoint
        # (system/tools breakpoints could still fire but neither is present)
        forwarded = mock_post.call_args.kwargs["json"]
        for msg in forwarded["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        assert "cache_control" not in blk

    @patch("httpx.AsyncClient.post")
    def test_cache_first_preserves_prefix_bytes(self, mock_post, client: TestClient):
        """The v0.3.1 fix: cache is applied BEFORE compression so the cached
        prefix stays byte-identical across turns. If we compressed first,
        tool-aware skeletonization would mutate the prefix every turn and
        Anthropic's exact-prefix cache would never hit."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"id": "m"}).encode()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_post.return_value = mock_response

        payload = {
            "model": "claude-sonnet-4-20250514",
            "system": "You are a helpful assistant.",
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object"},
                }
            ],
            "messages": [
                {"role": "user", "content": "fix app.py"},
                {"role": "assistant", "content": "Reading app.py..."},
                {"role": "user", "content": "what does it do?"},
                {"role": "assistant", "content": "It runs a web server."},
                {"role": "user", "content": "refund logic"},
                {"role": "assistant", "content": "def refund(order_id):\\n    pass"},
                {"role": "user", "content": "test it"},
            ],
        }

        response = client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        forwarded = mock_post.call_args.kwargs["json"]

        # Cache headers present
        assert int(response.headers["X-TokenTamer-Cache-Breakpoints"]) >= 1

        # The prefix message (index len-3 = 4) should have cache_control
        # because session cache ran on the original body before compression.
        prefix_msg = forwarded["messages"][4]
        has_cache = False
        content = prefix_msg.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and "cache_control" in blk:
                    has_cache = True
                    break
        assert has_cache, "Prefix message should have cache_control from cache-first"
