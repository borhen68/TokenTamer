"""
FastAPI proxy server for TokenTamer.

Intercepts LLM API requests (OpenAI and Anthropic format), compresses
background code blocks via AST skeletonization, and forwards the optimized
payload to the upstream provider. Responses are streamed back transparently.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator, List, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("token_guard")

from .config import Config
from .context_analyzer import ContextAnalyzer
from .dashboard import Dashboard
from .skeletonizer import Skeletonizer
from .token_counter import (
    FileStats,
    RequestMetrics,
    SessionMetrics,
    TokenCounter,
)
from . import upstream_resolver


def create_app(
    config: Config,
    metrics: SessionMetrics,
    dashboard: Optional[Dashboard] = None,
    ssl_mode: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Application configuration.
        metrics: Shared session metrics for the dashboard.
        dashboard: Optional dashboard instance for UI updates.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="TokenTamer Proxy",
        description="Smart context-aware token compactor for LLM coding agents",
        version="0.1.0",
        docs_url=None,  # Disable docs in production proxy
        redoc_url=None,
    )

    # Initialize components
    skeletonizer = Skeletonizer(
        keep_docstrings=config.skeletonizer.keep_docstrings,
        keep_class_attrs=config.skeletonizer.keep_class_attrs,
    )
    analyzer = ContextAnalyzer(skeletonizer, repo_path=config.repo_path)
    counter = TokenCounter()

    # Shared HTTP client (connection pooling)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )

    # In SSL interception mode, /etc/hosts points upstream domains to localhost.
    # We must bypass that mapping so httpx reaches the real APIs.
    if ssl_mode:
        import socket
        socket.getaddrinfo = upstream_resolver._patched_getaddrinfo
        logger.info("SSL interception mode: socket.getaddrinfo patched for upstream DNS bypass")

    @app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()

    # ──────────────────────────────────────────────────────────
    #  Health check
    # ──────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "requests_processed": metrics.total_requests,
            "tokens_saved": metrics.tokens_saved,
        }

    # ──────────────────────────────────────────────────────────
    #  OpenAI-compatible routes
    # ──────────────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request):
        """Intercept OpenAI Chat Completions, compress, and forward."""
        body = await request.json()
        headers = dict(request.headers)

        # Resolve API key
        auth_header = headers.get("authorization", "")
        api_key = ""
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        if not api_key:
            api_key = config.get_api_key("openai")

        model = body.get("model", "gpt-4o")
        messages = body.get("messages", [])
        is_streaming = body.get("stream", False)

        # ── Compress ──
        try:
            compressed_messages, analysis = analyzer.analyze_and_compress(messages)
            body["messages"] = compressed_messages
        except Exception as e:
            logger.warning(f"Compression failed: {e}. Forwarding original payload.")
            compressed_messages = messages
            analysis = None

        # ── Count tokens and record metrics ──
        original_tokens = counter.count_messages(messages, model)
        compressed_tokens = counter.count_messages(compressed_messages, model)

        file_stats = []
        if analysis:
            file_stats = [
                FileStats(
                    filename=block.filename or "unknown",
                    original_tokens=counter.count(block.content, model),
                    compressed_tokens=counter.count(
                        block.skeleton_result.skeleton if block.skeleton_result else block.content,
                        model,
                    ),
                    was_skeletonized=block.skeleton_result is not None and block.skeleton_result.was_compressed,
                )
                for block in analysis.code_blocks
            ]

        req_metrics = RequestMetrics(
            timestamp=time.time(),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            model=model,
            file_stats=file_stats,
        )

        # Estimate cost savings
        pricing = config.get_pricing(model)
        original_cost = counter.estimate_cost(original_tokens, 0, pricing.input, pricing.output)
        compressed_cost = counter.estimate_cost(compressed_tokens, 0, pricing.input, pricing.output)
        cost_saved = original_cost - compressed_cost

        metrics.record_request(req_metrics, cost_saved)

        # ── Forward to upstream ──
        upstream_url = f"{config.upstream.openai_url}/v1/chat/completions"
        forward_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if is_streaming:
            return StreamingResponse(
                _stream_proxy(http_client, upstream_url, forward_headers, body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-TokenTamer-Saved": str(req_metrics.tokens_saved),
                },
            )
        else:
            response = await http_client.post(
                upstream_url,
                headers=forward_headers,
                json=body,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    "Content-Type": response.headers.get("content-type", "application/json"),
                    "X-TokenTamer-Saved": str(req_metrics.tokens_saved),
                },
            )

    @app.post("/v1/completions")
    async def openai_completions(request: Request):
        """Pass-through for legacy completions (no compression for non-chat)."""
        body = await request.json()
        headers = dict(request.headers)

        auth_header = headers.get("authorization", "")
        api_key = auth_header[7:] if auth_header.startswith("Bearer ") else config.get_api_key("openai")

        upstream_url = f"{config.upstream.openai_url}/v1/completions"
        forward_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        is_streaming = body.get("stream", False)
        if is_streaming:
            return StreamingResponse(
                _stream_proxy(http_client, upstream_url, forward_headers, body),
                media_type="text/event-stream",
            )
        else:
            response = await http_client.post(upstream_url, headers=forward_headers, json=body)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={"Content-Type": response.headers.get("content-type", "application/json")},
            )

    @app.get("/v1/models")
    async def openai_models(request: Request):
        """Pass-through for model listing."""
        headers = dict(request.headers)
        auth_header = headers.get("authorization", "")
        api_key = auth_header[7:] if auth_header.startswith("Bearer ") else config.get_api_key("openai")

        upstream_url = f"{config.upstream.openai_url}/v1/models"
        response = await http_client.get(
            upstream_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={"Content-Type": response.headers.get("content-type", "application/json")},
        )

    # ──────────────────────────────────────────────────────────
    #  Anthropic-compatible routes
    # ──────────────────────────────────────────────────────────

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        """Intercept Anthropic Messages API, compress, and forward."""
        body = await request.json()
        headers = dict(request.headers)

        # Anthropic uses x-api-key header
        api_key = headers.get("x-api-key", "") or config.get_api_key("anthropic")
        anthropic_version = headers.get("anthropic-version", "2023-06-01")

        model = body.get("model", "claude-sonnet-4-20250514")
        messages = body.get("messages", [])
        is_streaming = body.get("stream", False)

        # Also check system prompt for file references
        system_content = body.get("system", "")
        all_messages = messages.copy()
        if system_content:
            if isinstance(system_content, str):
                all_messages.insert(0, {"role": "system", "content": system_content})
            elif isinstance(system_content, list):
                # Anthropic system can be a list of content blocks
                text_parts = " ".join(
                    p.get("text", "") for p in system_content if isinstance(p, dict)
                )
                all_messages.insert(0, {"role": "system", "content": text_parts})

        # ── Compress ──
        try:
            compressed_messages, analysis = analyzer.analyze_and_compress(
                messages, all_messages=all_messages
            )
            body["messages"] = compressed_messages
        except Exception as e:
            logger.warning(f"Compression failed: {e}. Forwarding original payload.")
            compressed_messages = messages
            analysis = None

        # ── Count tokens and record metrics ──
        original_tokens = counter.count_messages(messages, model)
        compressed_tokens = counter.count_messages(compressed_messages, model)

        file_stats = []
        if analysis:
            file_stats = [
                FileStats(
                    filename=block.filename or "unknown",
                    original_tokens=counter.count(block.content, model),
                    compressed_tokens=counter.count(
                        block.skeleton_result.skeleton if block.skeleton_result else block.content,
                        model,
                    ),
                    was_skeletonized=block.skeleton_result is not None and block.skeleton_result.was_compressed,
                )
                for block in analysis.code_blocks
            ]

        req_metrics = RequestMetrics(
            timestamp=time.time(),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            model=model,
            file_stats=file_stats,
        )

        pricing = config.get_pricing(model)
        original_cost = counter.estimate_cost(original_tokens, 0, pricing.input, pricing.output)
        compressed_cost = counter.estimate_cost(compressed_tokens, 0, pricing.input, pricing.output)
        cost_saved = original_cost - compressed_cost

        metrics.record_request(req_metrics, cost_saved)

        # ── Forward to upstream ──
        upstream_url = f"{config.upstream.anthropic_url}/v1/messages"
        forward_headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
        }

        # Forward any additional anthropic headers
        for key, value in headers.items():
            if key.startswith("anthropic-") and key != "anthropic-version":
                forward_headers[key] = value

        if is_streaming:
            return StreamingResponse(
                _stream_proxy(http_client, upstream_url, forward_headers, body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-TokenTamer-Saved": str(req_metrics.tokens_saved),
                },
            )
        else:
            response = await http_client.post(
                upstream_url,
                headers=forward_headers,
                json=body,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={
                    "Content-Type": response.headers.get("content-type", "application/json"),
                    "X-TokenTamer-Saved": str(req_metrics.tokens_saved),
                },
            )

    # ──────────────────────────────────────────────────────────
    #  Catch-all pass-through for unrecognized routes
    # ──────────────────────────────────────────────────────────

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(request: Request, path: str):
        """Pass through any unrecognized routes to the OpenAI upstream."""
        body = await request.body()
        headers = dict(request.headers)

        # Remove host header to avoid conflicts
        headers.pop("host", None)

        upstream_url = f"{config.upstream.openai_url}/{path}"

        response = await http_client.request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            content=body,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={"Content-Type": response.headers.get("content-type", "application/json")},
        )

    return app


async def _stream_proxy(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: dict,
) -> AsyncGenerator[bytes, None]:
    """
    Stream the response from the upstream API back to the caller.

    Yields raw bytes as they arrive from the upstream, preserving
    the SSE framing exactly as the provider sends it.
    """
    try:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk
    except httpx.ReadTimeout:
        # Send an error event if the upstream times out
        error = json.dumps({"error": {"message": "Upstream API timeout", "type": "timeout"}})
        yield f"data: {error}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except httpx.HTTPError as e:
        error = json.dumps({"error": {"message": str(e), "type": "proxy_error"}})
        yield f"data: {error}\n\n".encode()
        yield b"data: [DONE]\n\n"


def _compress_messages(
    messages: List[dict],
    model: str,
    analyzer: ContextAnalyzer,
    counter: TokenCounter,
    config: Config,
) -> RequestMetrics:
    """
    Analyze and compress messages. Returns metrics but note that
    the actual compression is done by analyzer.analyze_and_compress().
    This is a helper for the non-streaming path.
    """
    # This function is kept for potential future use
    return RequestMetrics(timestamp=time.time(), model=model)
