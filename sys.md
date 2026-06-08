Context & Role

You are an expert Python systems engineer. We are building "TokenTamer", a highly advanced Codebase Memory Layer and Context Assembler for AI coding agents.

We are NOT building a simple token-compression proxy. We are building a retrieval-augmented context engine that answers one fundamental question: "Given a specific coding task and a project repository, what exact context does an LLM need to successfully complete it?"

The Mission: Phase 1 (The Core Library)

Ignore proxies, APIs, FastAPIs, or terminal dashboards for now. Our immediate goal is to build the defensible core library: The Context Engine.

The Context Engine will take a target codebase path and a user query, and return a mathematically ranked list of the most relevant files/skeletons needed for the LLM prompt. It relies on three distinct signals to calculate relevance:

The Structural Signal (Dependency Graph): What imports what?

The Temporal Signal (Git Recency): What was edited recently?

The Intent Signal (Semantic Relevance): What files actually match the user's query?

Project Structure to Create

Please set up a Python project with the following structure:

token_tamer_core/
├── __init__.py
├── dependency_engine.py   # Parses AST to build an import graph
├── git_engine.py          # Uses git log to score file recency
├── semantic_engine.py     # Uses sentence-transformers for query-to-file matching
└── assembler.py           # The master weighting algorithm that combines the 3 signals


Your First Task

Please write the complete code for dependency_engine.py and git_engine.py.

Requirements for dependency_engine.py:

Use Python's built-in ast module.

It should scan a given directory for all .py files.

It must extract all import X and from Y import Z statements to build a directed graph (using a dictionary or networkx).

It should have a method: get_dependencies(filepath, depth=1) that returns all files the target file depends on, and files that depend on the target.

Requirements for git_engine.py:

Use Python's subprocess to run git log --name-only or use GitPython.

It should have a method: get_recency_scores(repo_path).

It should return a dictionary mapping file paths to a decay score (e.g., edited today = 1.0, edited 1 week ago = 0.8, edited 1 year ago = 0.1).

Write clean, modular, well-documented Python code with type hints. Start with these two files.