# 🚀 TokenTamer

[![CI](https://github.com/borhen68/TokenTamer/actions/workflows/ci.yml/badge.svg)](https://github.com/borhen68/TokenTamer/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A drop-in proxy that compresses bloated code context in real-time, cutting LLM API costs by 50–80% without losing what the model actually needs to know.**

TokenTamer is an intelligent, drop-in middleware proxy that sits between any AI coding agent (Aider, Cursor, Claude Code, Codex) and the LLM API. It intercepts raw payloads, dynamically parses the AST of code files, and compresses "background" files into structural skeletons — slashing token costs by up to 90%.

## ✨ Features

- **🔌 Drop-in proxy** — No changes needed to your coding agent. Just change the API base URL.
- **🧠 Smart active file detection** — Automatically identifies which files you're working on and leaves them 100% intact.
- **🌳 AST-based compression** — Strips function bodies while preserving signatures, imports, and class structures.
- **💰 Real-time cost tracking** — Beautiful terminal dashboard showing tokens saved and money saved.
- **🔄 Full streaming support** — Transparent SSE streaming for both OpenAI and Anthropic APIs.
- **⚡ Zero latency overhead** — Compression happens locally in milliseconds.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/borhen68/TokenTamer.git
cd TokenTamer

# Install with pip
pip install -e .
```

### Usage

```bash
# Start the proxy (default: http://127.0.0.1:8000)
token-tamer

# Or with custom settings
token-tamer --port 9000 --host 0.0.0.0

# With a custom config file
token-tamer --config /path/to/config.yaml

# Without the terminal dashboard
token-tamer --no-dashboard
```

### Configure Your Agent

Point your coding agent's API base URL to TokenTamer:

**Aider:**
```bash
aider --openai-api-base http://127.0.0.1:8000/v1
```

**Cursor:** Update the API base URL in Settings → Models → OpenAI API Base

**Claude Code / Codex CLI (hardcoded endpoints):**

These tools don't expose a base URL setting. Use **SSL interception mode**:

```bash
# 1. Generate certificates and print setup instructions
token-tamer --ssl --port 443

# 2. Trust the CA (macOS)
sudo security add-trusted-cert -d -r trustRoot \
  -k /Library/Keychains/System.keychain \
  ~/.config/token-tamer/certs/ca-cert.pem

# 3. Edit /etc/hosts (needs sudo)
sudo nano /etc/hosts
# Add these lines:
127.0.0.1 api.openai.com
127.0.0.1 api.anthropic.com

# 4. Now Claude Code and Codex CLI traffic flows through TokenTamer
claude "create a snake game"
codex "refactor this module"
```

TokenTamer intercepts the HTTPS traffic, compresses the context, and forwards
 to the real APIs while transparently bypassing your `/etc/hosts` mapping.

### API Keys

TokenTamer resolves API keys in this priority order:

1. **Request headers** — Keys sent by your agent (default behavior, zero config needed)
2. **Environment variables** — `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
3. **Config file** — `config.yaml`

## 📊 How It Works

```
Your Agent                    TokenTamer                      LLM API
    │                              │                              │
    │── 100k token payload ──────▶│                              │
    │                              │── Identify active files      │
    │                              │── Skeletonize background     │
    │                              │── 15k token payload ────────▶│
    │                              │                              │
    │                              │◀──── Streaming response ─────│
    │◀── Streaming response ──────│                              │
    │                              │                              │
    │                              │── Dashboard: saved $2.45! 💰 │
```

### Before (Heavy Token Cost):
```python
def calculate_tax(amount: float, region: str) -> float:
    """Calculates regional tax rates based on complex logic."""
    rate = get_base_rate(region)
    adjustments = fetch_adjustments(region, amount)
    if amount > THRESHOLD:
        rate *= 1.05
    # ... 50 more lines of complex math ...
    return final_tax
```

### After (Lightweight Skeleton):
```python
# [TOKEN-GUARD: Compressed — structural skeleton only]
def calculate_tax(amount: float, region: str) -> float: ...
```

The LLM still knows `calculate_tax` exists and how to call it, but doesn't waste tokens reading the implementation.

## ⚙️ Configuration

Create a `config.yaml` in your working directory:

```yaml
proxy:
  host: "127.0.0.1"
  port: 8000

upstream:
  openai_url: "https://api.openai.com"
  anthropic_url: "https://api.anthropic.com"

context:
  repo_path: "/path/to/your/codebase"  # Enables semantic active-file detection

skeletonizer:
  keep_docstrings: false      # Preserve function docstrings?
  keep_class_attrs: true      # Keep class-level attributes?

pricing:                       # Per 1M tokens for cost estimation
  gpt-4o:
    input: 2.50
    output: 10.00
  claude-sonnet-4-20250514:
    input: 3.00
    output: 15.00
```

## 🌐 Multi-Language Support

TokenTamer skeletonizes more than just Python:

| Language   | Method      | Status |
|------------|------------|--------|
| Python     | Native AST | ✅     |
| JavaScript | Brace-balance heuristic | ✅ |
| TypeScript | Brace-balance heuristic | ✅ |
| Go         | Brace-balance heuristic | ✅ |
| Rust       | Brace-balance heuristic | ✅ |
| Java / C# / C / C++ | Brace-balance heuristic | ✅ |

## 🧠 Semantic Active-File Detection

If you provide a `repo_path` in `config.yaml` and install `sentence-transformers`,
TokenTamer uses embeddings to detect which files are semantically relevant to your
query — even if you don't mention them by name.

```bash
pip install sentence-transformers scikit-learn
```

## 🛠 Supported APIs

| Provider  | Endpoint                  | Status |
|-----------|--------------------------|--------|
| OpenAI    | `/v1/chat/completions`   | ✅     |
| OpenAI    | `/v1/completions`        | ✅ (pass-through) |
| OpenAI    | `/v1/models`             | ✅ (pass-through) |
| Anthropic | `/v1/messages`           | ✅     |

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 📋 Requirements

- Python 3.10+
- Dependencies: FastAPI, uvicorn, httpx, tiktoken, rich, pyyaml

## 📜 License

MIT
