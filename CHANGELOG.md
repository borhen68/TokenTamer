# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2024-06-09

### Added
- **Tool-aware compression** — Claude Code and Codex CLI now get real savings.
  TokenTamer tracks `tool_use → file_path` mappings and skeletonizes stale
  `tool_result` file reads while preserving the most recent read of each file.
- New `/v1/responses` endpoint for the OpenAI Responses API (used by Codex CLI).
- `--passthrough` flag: kill-switch that disables all compression while still proxying.
- `--no-tool-compression` flag: disable smart tool-aware path only.
- **Long-lived session hijacking** — Exploits Anthropic prompt caching (`cache_control` breakpoints).
  Injected into outbound Anthropic requests: tools array, system prompt, and conversation
  prefix all get `cache_control` markers. Cached input tokens cost **$0.30/Mtoken** vs $3.00/Mtoken
  regular — up to a 90% discount on long Claude Code sessions.
- `--no-session-cache` flag to disable prompt caching injection.
- 20 unit + integration tests for session cache covering breakpoint placement, idempotency,
  session tracking, token estimation, and robustness against malformed payloads.
- 3 integration tests verifying server-level cache header injection and opt-out behavior.

### Changed
- Tool definitions, `tool_use` blocks, and the latest `tool_result` per file are
  never modified — this is now the default behavior, not a special "safety mode".

## [0.1.0] - 2024-06-08

### Added
- Initial release as **TokenTamer** (formerly Token-Guard)
- Drop-in HTTP/HTTPS proxy for OpenAI and Anthropic APIs
- AST-based Python code skeletonization (strip function bodies, keep signatures)
- Multi-language skeletonization for C-style languages (JS, TS, Go, Rust, Java, C#, C, C++)
- Smart active file detection via regex patterns and optional semantic relevance scoring
- SSL interception mode for hardcoded clients (Claude Code, Codex CLI)
- Real-time terminal dashboard with token savings and cost estimates
- Full SSE streaming support
- Configurable compression via YAML config
- Comprehensive pytest test suite with mocked upstream
