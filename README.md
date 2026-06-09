# 🚀 TokenTamer

[![CI](https://github.com/borhen68/TokenTamer/actions/workflows/ci.yml/badge.svg)](https://github.com/borhen68/TokenTamer/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A drop-in proxy that compresses bloated code context in real-time, cutting LLM API costs by 50–80% on plain-chat coding agents.**

TokenTamer is a middleware proxy that sits between an AI coding agent and the LLM API. It intercepts raw payloads, parses code with AST, and replaces "background" files with structural skeletons. The agent still sees signatures, classes, and imports — it just stops paying for function bodies it isn't editing.

> ⚠️ **Alpha software.** This is a real project in active development, not a polished SaaS. Please read the support matrix below before installing.

## 🧪 Support Status

| Client | HTTPS interception | Compression active | Notes |
|--------|--------------------|--------------------|-------|
| **Aider** (`--openai-api-base`) | ✅ Not needed | ✅ Full | Best supported. Use the proxy URL directly. |
| **Cursor** (custom base URL) | ✅ Not needed | ✅ Full | Best supported. |
| **Plain `curl` / SDK calls** | ✅ Not needed | ✅ Full | Great for testing. |
| **Claude Code** (hardcoded endpoint) | ✅ Works | ✅ Tool-aware | Stale file reads in `tool_result` get skeletonized; latest read stays intact. |
| **Codex CLI** (hardcoded endpoint) | ✅ Works | ✅ Tool-aware | Same engine via `/v1/responses`. |

**How tool-aware compression works.** Agents like Claude Code call `Read(file)` repeatedly. The conversation accumulates the same file dumped multiple times. TokenTamer tracks every `tool_use → file` mapping, then skeletonizes the *older* `tool_result` reads while keeping the **most recent** read of each file 100% intact. `tool_use` blocks and tool definitions are never touched.

If something ever breaks, hit the kill switch:
```bash
token-tamer --ssl --port 443 --passthrough            # disable all compression
# or
token-tamer --ssl --port 443 --no-tool-compression    # disable only tool-aware path
```

## 🚨 Known Limitations

- **Compression depends on re-reads.** Single-read sessions get no tool savings (just text compression). Long sessions where the agent re-reads files benefit the most.
- **Heuristic file detection.** We look for `file_path` / `path` / `filename` keys in tool inputs. Exotic agents with unusual schemas may be missed.
- **Multi-turn cross-request caching** is not yet implemented.
- **macOS only** for the one-line cert setup. Linux/Windows users need to trust the CA manually.
- **No production benchmarks yet.** Savings numbers come from unit tests with synthetic payloads, not real long Claude Code sessions.

## 🗺️ Roadmap

- [x] **v0.2** — Tool-aware compression (✅ shipped)
- [ ] **v0.3** — Multi-turn / cross-request cache so repeat content isn't re-sent
- [ ] **v0.4** — Tree-sitter for proper multi-language AST (current C-style support is a brace-balance heuristic)
- [ ] **v0.5** — Web dashboard with per-file compression heatmap

## ✨ Features

- **🔌 Drop-in proxy** — No changes needed to your coding agent. Just change the API base URL.
- **🧠 Smart active file detection** — Automatically identifies which files you're working on and leaves them 100% intact.
- **🌳 AST-based compression** — Strips function bodies while preserving signatures, imports, and class structures.
- **💰 Real-time cost tracking** — Beautiful terminal dashboard showing tokens saved and money saved.
- **🔄 Full streaming support** — Transparent SSE streaming for both OpenAI and Anthropic APIs.
- **⚡ Zero latency overhead** — Compression happens locally in milliseconds.

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- Python **3.9 or newer** (`python3 --version`)
- macOS, Linux, or Windows (Windows = manual cert trust step)
- `openssl` (pre-installed on macOS & most Linux)

### 1. Install

```bash
git clone https://github.com/borhen68/TokenTamer.git
cd TokenTamer

# Recommended: use a virtual environment to avoid messing with system Python
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install -e .
```

Verify it installed:
```bash
token-tamer --version
# → TokenTamer 0.2.0
```

### 2. Choose Your Path

**👉 Path A — Aider, Cursor, or your own SDK code** (no SSL setup needed):
```bash
token-tamer --port 8000 --no-dashboard
```
Then point your tool's API base URL at `http://127.0.0.1:8000/v1`:
```bash
aider --openai-api-base http://127.0.0.1:8000/v1
```
For Cursor: Settings → Models → OpenAI API Base → `http://127.0.0.1:8000/v1`. **Done.** ✅

**👉 Path B — Claude Code or Codex CLI** (SSL setup, one-time):

These tools hardcode the API URL. We use HTTPS interception:

```bash
# Step 1 — Generate the local certificate (just runs and exits)
token-tamer --ssl --port 8443 --no-dashboard &
sleep 2 && kill %1

# Step 2 — Trust the certificate (macOS)
sudo security add-trusted-cert -d -r trustRoot \
  -k /Library/Keychains/System.keychain \
  ~/.config/token-tamer/certs/ca-cert.pem

# Step 3 — Redirect API domains to localhost
echo "127.0.0.1 api.openai.com"     | sudo tee -a /etc/hosts
echo "127.0.0.1 api.anthropic.com"  | sudo tee -a /etc/hosts

# Step 4 — Run TokenTamer on port 443 (sudo required for low ports)
sudo $(which token-tamer) --ssl --port 443 --no-dashboard
```

Leave that terminal open, then in a **new terminal**:
```bash
claude "create a snake game"     # or
codex "refactor this module"
```

You're now intercepting + compressing. 🎉

### 3. Verify It's Working

```bash
# Path A check:
curl http://127.0.0.1:8000/health

# Path B check:
curl https://api.openai.com/health    # Should return TokenTamer's JSON, not OpenAI's
```

Both should return:
```json
{"status":"ok","version":"0.2.0","requests_processed":0,"tokens_saved":0}
```

### 4. Cleanup (Uninstall)

```bash
# Remove /etc/hosts entries
sudo sed -i.bak '/api.openai.com/d;/api.anthropic.com/d' /etc/hosts

# Untrust the cert
sudo security remove-trusted-cert -d ~/.config/token-tamer/certs/ca-cert.pem

# Uninstall the package
pip uninstall token-tamer
```

### 🆘 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `command not found: token-tamer` | Activate your venv: `source venv/bin/activate` |
| `ModuleNotFoundError: No module named 'uvicorn'` | Same — venv not active |
| `address already in use` on port 8000 | `lsof -ti :8000 \| xargs kill -9` |
| `Permission denied` on port 443 | Use `sudo` for ports <1024, or pick a higher port |
| `SSL certificate problem` from `curl` | Re-run the `security add-trusted-cert` step, then open a NEW terminal |
| Claude Code hangs / errors | Hit the kill switch: restart with `--passthrough` |
| Compression broke something | Restart with `--no-tool-compression` and [file an issue](https://github.com/borhen68/TokenTamer/issues) |

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
