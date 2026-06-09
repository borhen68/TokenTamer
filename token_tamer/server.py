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

logger = logging.getLogger("token_tamer")

from . import __version__
from .config import Config
from .context_analyzer import ContextAnalyzer
from .dashboard import Dashboard
from .session_cache import SessionCache
from .skeletonizer import Skeletonizer
from .token_counter import (
    FileStats,
    RequestMetrics,
    SessionMetrics,
    TokenCounter,
)
from . import upstream_resolver


def _request_has_tools(body: dict) -> bool:
    """Detect if the request involves tool/function calling.

    When tools are present, compression is risky because the model may
    need exact code context to generate correct tool arguments. We bail
    out to safe pass-through behavior in this case.
    """
    if not isinstance(body, dict):
        return False
    # OpenAI: tools=[{...}] or functions=[{...}]
    if body.get("tools") or body.get("functions"):
        return True
    if body.get("tool_choice") not in (None, "none"):
        return True
    # Anthropic: tools=[{...}]
    # (Same key, already covered above.)
    # Tool result messages embedded in history
    for msg in body.get("messages", []) or []:
        if isinstance(msg, dict):
            if msg.get("role") in ("tool", "function"):
                return True
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in (
                        "tool_use", "tool_result",
                    ):
                        return True
    return False


def create_app(
    config: Config,
    metrics: SessionMetrics,
    dashboard: Optional[Dashboard] = None,
    ssl_mode: bool = False,
    passthrough: bool = False,
    compress_with_tools: bool = True,
    session_cache_enabled: bool = True,
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
        version=__version__,
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
    session_cache = SessionCache()

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
            "version": __version__,
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

        # ── Compress (tool-aware) ──
        compressed_messages = messages
        analysis = None
        has_tools = _request_has_tools(body)
        if passthrough:
            pass  # No compression at all
        elif has_tools and not compress_with_tools:
            pass  # Tools present + smart compression disabled → full passthrough
        else:
            try:
                if has_tools:
                    compressed_messages, analysis = analyzer.analyze_and_compress_tool_aware(
                        messages
                    )
                else:
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

        # ── Compress (tool-aware) ──
        compressed_messages = messages
        analysis = None
        has_tools = _request_has_tools(body)
        if passthrough:
            pass
        elif has_tools and not compress_with_tools:
            pass
        else:
            try:
                if has_tools:
                    compressed_messages, analysis = analyzer.analyze_and_compress_tool_aware(
                        messages, all_messages=all_messages
                    )
                else:
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

        # ── Long-lived session hijacking via prompt caching ──
        cache_info = {"breakpoints": 0, "cached_tokens_estimate": 0}
        if session_cache_enabled and not passthrough:
            try:
                body, cache_info = session_cache.apply(body)
                if cache_info["breakpoints"] > 0:
                    logger.info(
                        f"Session cache: session={cache_info['session_id']} "
                        f"turn={cache_info['turn_count']} "
                        f"breakpoints={cache_info['breakpoints']} "
                        f"cached_tokens≈{cache_info['cached_tokens_estimate']}"
                    )
            except Exception as e:
                logger.warning(f"Session cache injection failed: {e}. Forwarding without it.")

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

        cache_headers = {
            "X-TokenTamer-Cache-Breakpoints": str(cache_info["breakpoints"]),
            "X-TokenTamer-Cache-Tokens": str(cache_info["cached_tokens_estimate"]),
        }
        if is_streaming:
            return StreamingResponse(
                _stream_proxy(http_client, upstream_url, forward_headers, body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-TokenTamer-Saved": str(req_metrics.tokens_saved),
                    **cache_headers,
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
                    **cache_headers,
                },
            )

    # ──────────────────────────────────────────────────────────
    #  OpenAI Responses API (used by Codex CLI)
    # ──────────────────────────────────────────────────────────

    @app.post("/v1/responses")
    async def openai_responses(request: Request):
        """Intercept OpenAI Responses API (Codex CLI).

        Codex sends `input` (string or list of message objects) instead of
        `messages`. We compress textual code blocks in user messages only,
        leaving tool calls and reasoning items untouched.
        """
        body = await request.json()
        headers = dict(request.headers)

        auth_header = headers.get("authorization", "")
        api_key = auth_header[7:] if auth_header.startswith("Bearer ") else config.get_api_key("openai")

        model = body.get("model", "gpt-4o")
        is_streaming = body.get("stream", False)
        raw_input = body.get("input", "")

        # Normalize input → list of pseudo-messages for the analyzer.
        # The Responses API accepts either a string or a list of items.
        synthesized: List[dict] = []
        if isinstance(raw_input, str):
            synthesized = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    item_type = item.get("type")
                    # Skip tool/reasoning items entirely
                    if item_type in ("function_call", "function_call_output",
                                       "tool_call", "tool_result", "reasoning"):
                        continue
                    content = item.get("content", "")
                    synthesized.append({"role": role, "content": content})

        has_tools = _request_has_tools(body)
        skip_compression = passthrough or (has_tools and not compress_with_tools)

        compressed_input = raw_input
        analysis = None
        if not skip_compression and synthesized:
            try:
                new_msgs, analysis = analyzer.analyze_and_compress(synthesized)
                # Rebuild input preserving structure
                if isinstance(raw_input, str) and new_msgs:
                    first = new_msgs[0].get("content", raw_input)
                    if isinstance(first, str):
                        compressed_input = first
                elif isinstance(raw_input, list):
                    compressed_input = _rewrite_responses_input(raw_input, new_msgs)
                body["input"] = compressed_input
            except Exception as e:
                logger.warning(f"Responses API compression failed: {e}. Forwarding original.")

        # Best-effort metrics
        try:
            original_tokens = counter.count_messages(synthesized, model)
            compressed_tokens = counter.count_messages(
                _coerce_to_messages(compressed_input), model
            )
        except Exception:
            original_tokens = compressed_tokens = 0

        req_metrics = RequestMetrics(
            timestamp=time.time(),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            model=model,
            file_stats=[],
        )
        pricing = config.get_pricing(model)
        original_cost = counter.estimate_cost(original_tokens, 0, pricing.input, pricing.output)
        compressed_cost = counter.estimate_cost(compressed_tokens, 0, pricing.input, pricing.output)
        metrics.record_request(req_metrics, original_cost - compressed_cost)

        upstream_url = f"{config.upstream.openai_url}/v1/responses"
        forward_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # Forward OpenAI-Beta and other custom headers
        for k, v in headers.items():
            if k.lower().startswith("openai-"):
                forward_headers[k] = v

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
                upstream_url, headers=forward_headers, json=body,
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


def _rewrite_responses_input(
    raw_input: list,
    compressed_messages: List[dict],
) -> list:
    """Re-merge compressed user/system content back into a Responses API `input` list.

    We only replace text content for items whose role matches one we compressed,
    preserving order. Tool/reasoning items pass through unchanged.
    """
    msg_iter = iter(compressed_messages)
    out: list = []
    for item in raw_input:
        if not isinstance(item, dict):
            out.append(item)
            continue
        item_type = item.get("type")
        if item_type in (
            "function_call", "function_call_output",
            "tool_call", "tool_result", "reasoning",
        ):
            out.append(item)
            continue
        try:
            replacement = next(msg_iter)
        except StopIteration:
            out.append(item)
            continue
        new_item = dict(item)
        new_item["content"] = replacement.get("content", item.get("content"))
        out.append(new_item)
    return out


def _coerce_to_messages(value) -> List[dict]:
    """Coerce a Responses API `input` value into a flat messages list for token counting."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        msgs: List[dict] = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                msgs.append({
                    "role": item.get("role", "user"),
                    "content": item.get("content", ""),
                })
        return msgs
    return []
