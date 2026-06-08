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
        assert data["version"] == "0.1.0"
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
