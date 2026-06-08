🚀 TokenTamer: The Smart Context-Aware Token Compactor

💡 The Core Problem

LLM coding agents (like Claude Code, Aider, and Cursor) are incredibly expensive to run on large codebases. When an agent needs to understand a project, it pulls in massive files—burning tens of thousands of tokens. Often, 90% of that code (deep inner logic, loops, docstrings) is irrelevant to the specific bug being fixed. This bloats the context window, triggers API rate limits, and costs developers thousands of dollars.

🛠 The Solution: TokenTamer

TokenTamer is an intelligent, drop-in middleware proxy that sits between any AI coding agent and the LLM API (OpenAI or Anthropic).

It intercepts the raw payload, dynamically parses the Abstract Syntax Tree (AST) of the codebase, and compresses "background" files into structural "skeletons" (just class and function signatures). It feeds the LLM only the deep logic it actually needs, triggering prompt caching and slashing costs by up to 90%.

🏗 System Architecture

TokenTamer operates entirely locally, functioning as an API interceptor. It does not require any changes to the underlying coding agent.

graph TD
    subgraph Local Machine
        A[Coding Agent\n(Aider, Cursor, Claude Code)] -->|1. Sends raw 100k Token payload| B(TokenTamer Proxy\nFastAPI / Go)
        B -->|2. Identifies Code Blocks| C{Context Analyzer}
        
        C -->|Active File (Mentioned in Prompt)| D[Leave 100% Intact]
        C -->|Background File| E[AST Skeletonizer]
        
        E -->|Strips Function Bodies| F[Compressed Structural Skeletons]
        D --> G(Reassembled Payload\n15k Tokens)
        F --> G
    end
    
    G -->|3. Forwards Optimized Payload| H[LLM API\nOpenAI / Anthropic]
    H -->|4. Streams Response| B
    B -->|5. Returns to Agent| A
    B -.->|6. Updates Dashboard| I[Real-time Terminal UI\nTokens & $$$ Saved]


⚙️ How the "Smart" Engineering Works

1. The Interceptor (API Proxy)

The user changes their API Base URL in their agent to point to localhost:8000. TokenTamer catches the POST /v1/chat/completions or Anthropic /v1/messages request.

2. The "Active File" Rule (Crucial for preventing AI breakage)

If you blindly compress everything, the AI can't fix bugs because it can't see the inner logic. TokenTamer is smart:

It scans the final user prompt (e.g., "Fix the math error in payment.py").

It parses the massive context payload.

If a code block belongs to payment.py, it leaves it 100% intact.

If a code block belongs to database.py or utils.py, it routes it to the Skeletonizer.

3. AST Pruning (The Skeletonizer)

Using native parsing libraries (like Python's ast or tree-sitter for multi-language support), it collapses massive files into structural maps.

Before (Heavy Token Cost):

def calculate_tax(amount: float, region: str) -> float:
    """Calculates regional tax rates based on complex logic."""
    # 50 lines of complex math, database lookups, 
    # and error handling here...
    return final_tax


After (Lightweight Context Map):

# [TOKEN-GUARD: Background file compressed]
def calculate_tax(amount: float, region: str) -> float: ...


The LLM still knows calculate_tax exists and how to call it, but doesn't waste tokens reading the math.

4. Forcing Prompt Caching

By keeping the structural skeletons highly deterministic (removing whitespace, standardizing format) and placing them at the start of the prompt, TokenTamer forces Anthropic's Prompt Caching to trigger. This drops the cost of reading the project architecture by another 80%.

5. The Terminal Flex (TUI)

While the agent runs, TokenTamer displays a rich terminal dashboard showing real-time metrics. Developers love transparency.

╭────────────────────── 🚀 TokenTamer Active ──────────────────────╮
│ 🟢 File: payment.py (Intact)                                      │
│ 🟡 File: database.py (Skeletonized - saved 4,200 tokens)          │
│ 💰 Session Savings: $2.45 | Context Reduction: 82.4%              │
╰───────────────────────────────────────────────────────────────────╯


🛠 Tech Stack Recommendations for the LLM

To build this robustly and quickly, ask the LLM to use:

Core Server: FastAPI (Python) or Go (for maximum speed and easy single-binary distribution).

Parsing: Python's native ast module (MVP) or tree-sitter (for V2 multi-language support).

Terminal UI: Rich (Python) or Bubbletea (Go).

Token Counting: tiktoken (standard OpenAI estimation).

🚀 Step-by-Step Build Plan for Your LLM

Build the Proxy Skeleton: Start with a basic FastAPI server that can catch a request, print it, and forward it to OpenAI via httpx, successfully streaming the response back.

Build the AST Parser: Create a function that takes a raw string of Python code and returns the def ...: ... skeleton.

Implement the "Active File" Regex: Create logic to detect code blocks in the payload and determine if they should be compressed based on the user's prompt.

Wire it Together: Intercept the payload, compress the background blocks, calculate token diffs, and forward the optimized payload.

Build the TUI: Add the Rich console panel to output the savings metrics.