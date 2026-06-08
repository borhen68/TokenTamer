"""
Token-Guard CLI entry point.

Usage:
    python -m token_guard [--port PORT] [--host HOST] [--config PATH]
    token-guard [--port PORT] [--host HOST] [--config PATH]
"""

from __future__ import annotations

import argparse
import signal
import sys

import uvicorn

from . import __version__, __app_name__
from .config import load_config
from .dashboard import Dashboard
from .server import create_app
from .token_counter import SessionMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="token-guard",
        description="🚀 Token-Guard: Smart context-aware token compactor for LLM coding agents",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to listen on (default: 8000, or from config)",
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default=None,
        help="Host to bind to (default: 127.0.0.1, or from config)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the terminal dashboard",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"{__app_name__} {__version__}",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for Token-Guard."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI args override config file
    host = args.host or config.proxy.host
    port = args.port or config.proxy.port

    # Create shared metrics
    metrics = SessionMetrics()

    # Create dashboard
    dashboard: Dashboard | None = None
    if not args.no_dashboard:
        dashboard = Dashboard(metrics, host=host, port=port)

    # Create the FastAPI app
    app = create_app(config, metrics, dashboard)

    # Print startup banner
    if dashboard:
        dashboard.print_startup_banner()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        if dashboard:
            dashboard.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the dashboard
    if dashboard:
        dashboard.start()

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",  # Keep logs minimal — dashboard shows what matters
        access_log=False,
    )


if __name__ == "__main__":
    main()
