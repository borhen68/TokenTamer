# Changelog

All notable changes to this project will be documented in this file.

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
