"""
FastAPI proxy server for TokenTamer.

Intercepts LLM API requests (OpenAI and Anthropic format), compresses
background code blocks via AST skeletonization, and forwards the optimized
payload to the upstream provider. Responses are streamed back transparently.
"""

from __future__ import annotations

import copy
import gzip
import io
import json
import logging
import time
import zlib
from typing import AsyncGenerator, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import zstandard as _zstd
except ImportError:  # pragma: no cover
    _zstd = None

try:
    import brotli as _brotli
except ImportError:  # pragma: no cover
    _brotli = None

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


def _anthropic_forward_headers(headers: dict, config: Config) -> dict:
    """Build upstream headers for Anthropic-compatible requests.

    Claude Code can authenticate with either API keys (`x-api-key`) or a
    Claude.ai OAuth/subscription token (`Authorization: Bearer ...`). Preserve
    whichever credential the client sent instead of forcing all requests into
    the API-key header shape.
    """
    forward_headers = {
        "Content-Type": "application/json",
        "anthropic-version": headers.get("anthropic-version", "2023-06-01"),
    }

    auth_header = headers.get("authorization", "")
    api_key = headers.get("x-api-key", "")
    if auth_header:
        forward_headers["Authorization"] = auth_header
    elif api_key:
        forward_headers["x-api-key"] = api_key
    else:
        configured_key = config.get_api_key("anthropic")
        if configured_key:
            forward_headers["x-api-key"] = configured_key

    # Forward Claude/Anthropic feature headers such as anthropic-beta.
    for key, value in headers.items():
        if key.startswith("anthropic-") and key != "anthropic-version":
            forward_headers[key] = value

    return forward_headers


def _with_query(url: str, request: Request) -> str:
    """Append the original query string when proxying a route."""
    query = request.url.query
    return f"{url}?{query}" if query else url


# Headers Codex CLI sends when authenticated with a ChatGPT subscription.
# `chatgpt-account-id` is the load-bearing one — its presence is how we detect
# subscription mode and how the upstream identifies which account to bill.
_CHATGPT_PASSTHROUGH_HEADERS = (
    "chatgpt-account-id",
    "originator",
    "session_id",
    "version",
)

# Hop-by-hop / connection-scoped headers we must NOT forward. Content-Length is
# stripped because we re-serialize the body; Accept-Encoding because we want
# raw bytes (especially for SSE streaming).
_HEADERS_TO_STRIP = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "accept-encoding",
    # We decompress incoming bodies and forward as plain bytes, so the
    # upstream must not be told the body is still compressed.
    "content-encoding",
}


def _decode_body(raw: bytes, encoding: str) -> bytes:
    """Decompress a request body according to its Content-Encoding header.

    Codex CLI sends `Content-Encoding: zstd` (and historically gzip), so we
    must decompress before JSON-parsing. Unknown encodings raise; callers turn
    that into a 415.
    """
    enc = (encoding or "").lower().strip()
    if not enc or enc == "identity":
        return raw
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    if enc == "zstd":
        if _zstd is None:
            raise RuntimeError("zstandard package not installed")
        # Codex CLI emits streaming zstd frames without a Content-Size header,
        # so plain `decompress(raw)` fails with "could not determine content
        # size". The streaming reader handles both framed and sized inputs.
        dctx = _zstd.ZstdDecompressor()
        with dctx.stream_reader(io.BytesIO(raw)) as reader:
            return reader.read()
    if enc == "br":
        if _brotli is None:
            raise RuntimeError("brotli package not installed")
        return _brotli.decompress(raw)
    raise ValueError(f"Unsupported Content-Encoding: {enc!r}")


async def _read_json(request: Request) -> dict:
    """Read and JSON-parse a request body, transparently handling compression."""
    raw = await request.body()
    encoding = request.headers.get("content-encoding", "")
    try:
        decoded = _decode_body(raw, encoding)
    except Exception as e:
        logger.warning(f"Failed to decode {encoding!r} body: {e}")
        raise HTTPException(status_code=415, detail=f"Cannot decode body: {e}")
    if not decoded:
        return {}
    return json.loads(decoded)


def _is_chatgpt_subscription_request(headers: dict) -> bool:
    """Detect a Codex CLI request authenticated via ChatGPT subscription.

    Codex always sends `chatgpt-account-id` in this mode and never in API-key
    mode, so we use it as the routing signal.
    """
    return bool(headers.get("chatgpt-account-id"))


def _openai_upstream_base(headers: dict, config: Config) -> str:
    """Pick the correct upstream base URL for an OpenAI-format request.

    ChatGPT-subscription Codex traffic must go to the ChatGPT backend, not the
    standard OpenAI API; everything else (API-key Codex, ChatCompletions
    clients) uses the configured OpenAI URL.
    """
    if _is_chatgpt_subscription_request(headers):
        return config.upstream.chatgpt_backend_url
    return config.upstream.openai_url


def _openai_forward_headers(headers: dict, config: Config) -> dict:
    """Build upstream headers for OpenAI-compatible requests.

    Forwards all client headers verbatim except hop-by-hop and connection-scoped
    ones (see `_HEADERS_TO_STRIP`). Preserving `User-Agent`, `OpenAI-Beta`,
    Codex-specific headers, etc. is required because the ChatGPT backend
    fingerprints on them and will throttle requests that look unrecognized.
    """
    forward_headers: dict = {
        k: v for k, v in headers.items() if k.lower() not in _HEADERS_TO_STRIP
    }
    forward_headers["Content-Type"] = "application/json"

    if not headers.get("authorization"):
        configured_key = config.get_api_key("openai")
        if configured_key:
            forward_headers["Authorization"] = f"Bearer {configured_key}"

    return forward_headers


def _openai_upstream_path(headers: dict, path: str) -> str:
    """Adapt an incoming OpenAI-style path to the upstream's path scheme.

    The OpenAI public API uses `/v1/...` (e.g., `/v1/responses`). The ChatGPT
    backend (`chatgpt.com/backend-api/codex`) drops the `/v1` and serves
    `/responses` directly, so a request proxied as-is hits the wrong URL and
    chatgpt.com returns 403. Strip the `/v1` prefix when routing to that
    backend; leave OpenAI-direct traffic untouched.
    """
    if _is_chatgpt_subscription_request(headers):
        if path.startswith("/v1/"):
            return path[3:]
        if path.startswith("v1/"):
            return path[2:]
    return path


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
        body = await _read_json(request)
        headers = dict(request.headers)

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
        upstream_url = _with_query(
            f"{_openai_upstream_base(headers, config)}"
            f"{_openai_upstream_path(headers, '/v1/chat/completions')}",
            request,
        )
        forward_headers = _openai_forward_headers(headers, config)

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
        body = await _read_json(request)
        headers = dict(request.headers)

        upstream_url = _with_query(
            f"{_openai_upstream_base(headers, config)}"
            f"{_openai_upstream_path(headers, '/v1/completions')}",
            request,
        )
        forward_headers = _openai_forward_headers(headers, config)

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

        upstream_url = _with_query(
            f"{_openai_upstream_base(headers, config)}"
            f"{_openai_upstream_path(headers, '/v1/models')}",
            request,
        )
        response = await http_client.get(
            upstream_url,
            headers=_openai_forward_headers(headers, config),
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
        body = await _read_json(request)
        headers = dict(request.headers)

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

        # ── Long-lived session hijacking: apply FIRST on original body ──
        # We must cache BEFORE compression because tool-aware compression is
        # stateful: a message that is "latest" on turn N becomes "stale" on
        # turn N+1 and gets skeletonized differently. If we compress first,
        # the prefix bytes mutate every turn → cache never hits → we pay the
        # more expensive cache WRITE cost ($3.75/M) instead of cache READ
        # ($0.30/M).  See README "Cache-First Design" section.
        cache_info = {"breakpoints": 0, "cached_tokens_estimate": 0, "prefix_end_index": -1}
        if session_cache_enabled and not passthrough:
            try:
                body, cache_info = session_cache.apply(body)
                if cache_info["breakpoints"] > 0:
                    logger.info(
                        f"Session cache: session={cache_info['session_id']} "
                        f"turn={cache_info['turn_count']} "
                        f"breakpoints={cache_info['breakpoints']} "
                        f"cached_tokens≈{cache_info['cached_tokens_estimate']} "
                        f"prefix_end={cache_info.get('prefix_end_index', -1)}"
                    )
            except Exception as e:
                logger.warning(f"Session cache injection failed: {e}. Forwarding without it.")

        # ── Compress (tool-aware) ──
        # We compress the FULL conversation for correct stale-read detection,
        # but only apply compression to messages AFTER the cached prefix.
        # Messages inside the cached prefix stay byte-for-byte identical so
        # Anthropic's exact-prefix cache actually hits on subsequent turns.
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
                    full_compressed, analysis = analyzer.analyze_and_compress_tool_aware(
                        messages, all_messages=all_messages
                    )
                else:
                    full_compressed, analysis = analyzer.analyze_and_compress(
                        messages, all_messages=all_messages
                    )
                # Merge: keep cached prefix intact, compress tail only
                prefix_end = cache_info.get("prefix_end_index", -1)
                if prefix_end >= 0 and prefix_end + 1 < len(messages):
                    cached_prefix = body.get("messages", messages)[: prefix_end + 1]
                    compressed_tail = full_compressed[prefix_end + 1 :]
                    compressed_messages = cached_prefix + compressed_tail
                else:
                    compressed_messages = full_compressed
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
        upstream_url = _with_query(f"{config.upstream.anthropic_url}/v1/messages", request)
        forward_headers = _anthropic_forward_headers(headers, config)

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

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        """Pass through Anthropic token counting requests.

        Claude Code calls this endpoint when using Anthropic Messages format
        gateways. Token counting must see the original request body, so do not
        compress or mutate it here.
        """
        body = await _read_json(request)
        headers = dict(request.headers)

        upstream_url = _with_query(
            f"{config.upstream.anthropic_url}/v1/messages/count_tokens",
            request,
        )
        response = await http_client.post(
            upstream_url,
            headers=_anthropic_forward_headers(headers, config),
            json=body,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={"Content-Type": response.headers.get("content-type", "application/json")},
        )

    # ──────────────────────────────────────────────────────────
    #  OpenAI Responses API (used by Codex CLI)
    # ──────────────────────────────────────────────────────────

    @app.get("/v1/responses")
    async def openai_responses_reject_upgrade(request: Request):
        """Reject WebSocket upgrade attempts cleanly.

        Codex CLI tries `wss://.../v1/responses` first, falling back to plain
        HTTPS POST on rejection. Without this handler the GET would fall to the
        catch-all, which forwards to chatgpt.com — wasting a round-trip and
        polluting logs with 403s from the upstream's branded error page.
        """
        return Response(status_code=426, content="WebSocket not supported")

    @app.post("/v1/responses")
    async def openai_responses(request: Request):
        """Intercept OpenAI Responses API (Codex CLI).

        Codex sends `input` (string or list of message objects) instead of
        `messages`. We compress textual code blocks in user messages only,
        leaving tool calls and reasoning items untouched.
        """
        body = await _read_json(request)
        headers = dict(request.headers)

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

        upstream_url = _with_query(
            f"{_openai_upstream_base(headers, config)}"
            f"{_openai_upstream_path(headers, '/v1/responses')}",
            request,
        )
        forward_headers = _openai_forward_headers(headers, config)

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
        """Pass through unrecognized routes to the right OpenAI-side upstream.

        Codex CLI on a ChatGPT subscription occasionally hits auxiliary routes
        (e.g. account info) on the ChatGPT backend; route those there based on
        the same `chatgpt-account-id` signal we use for /v1/responses.
        """
        body = await request.body()
        headers = dict(request.headers)

        # Remove host header to avoid conflicts
        headers.pop("host", None)

        upstream_url = _with_query(
            f"{_openai_upstream_base(headers, config)}"
            f"{_openai_upstream_path(headers, '/' + path)}",
            request,
        )

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
