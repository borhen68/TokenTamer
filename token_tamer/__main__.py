"""
TokenTamer CLI entry point.

Usage:
    python -m token_tamer [--port PORT] [--host HOST] [--config PATH]
    token-tamer [--port PORT] [--host HOST] [--config PATH]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

import uvicorn

from . import __version__, __app_name__
from .cert_manager import CertManager
from .config import load_config
from .dashboard import Dashboard
from .server import create_app
from .token_counter import SessionMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="token-tamer",
        description="🚀 TokenTamer: Drop-in proxy that compresses bloated code context, cutting LLM API costs by 50–80%.",
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
        "--ssl",
        action="store_true",
        help=(
            "Enable HTTPS interception mode for Claude Code / Codex CLI. "
            "Requires trusting the CA and editing /etc/hosts."
        ),
    )
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help=(
            "Kill-switch: disable all compression. Still proxies and records "
            "metrics. Use this if an agent breaks and you need a safe fallback."
        ),
    )
    parser.add_argument(
        "--no-tool-compression",
        action="store_true",
        help=(
            "Disable smart tool-aware compression. When tools are detected, "
            "forward the request untouched. Use this if smart compression "
            "ever breaks an agent (please file a bug too)."
        ),
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"{__app_name__} {__version__}",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for TokenTamer."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load configuration
    config = load_config(args.config)

    # CLI args override config file
    host = args.host or config.proxy.host
    port = args.port or config.proxy.port

    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    if args.ssl:
        cert_mgr = CertManager()
        cert, key = cert_mgr.ensure_server_cert()
        ssl_certfile = str(cert)
        ssl_keyfile = str(key)
        cert_mgr.print_instructions()

        # Suggest privileged port if not already set
        if port == 8000:
            print("💡 Hint: Use --port 443 for full interception (requires sudo)")
            print("   or --port 8443 and configure clients manually.\n")

    # Create shared metrics
    metrics = SessionMetrics()

    # Create dashboard
    dashboard: Dashboard | None = None
    if not args.no_dashboard:
        dashboard = Dashboard(metrics, host=host, port=port)

    # Create the FastAPI app
    app = create_app(
        config,
        metrics,
        dashboard,
        ssl_mode=args.ssl,
        passthrough=args.passthrough,
        compress_with_tools=not args.no_tool_compression,
    )

    if args.passthrough:
        logging.getLogger("token_tamer").warning(
            "⚠️  Passthrough mode enabled — no compression will be applied."
        )

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
    uvicorn_kwargs = dict(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    if ssl_certfile and ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
