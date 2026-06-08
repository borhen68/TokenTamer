"""
Rich terminal dashboard for Token-Guard.

Displays a beautiful real-time terminal UI showing proxy status,
per-file compression results, and cumulative session savings.
Runs in a background thread alongside the FastAPI server.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

from .token_counter import SessionMetrics, RequestMetrics, FileStats


def _format_uptime(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_tokens(count: int) -> str:
    """Format token count with comma separators."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def _format_money(amount: float) -> str:
    """Format a dollar amount."""
    return f"${amount:.2f}"


def _compression_bar(ratio: float, width: int = 20) -> Text:
    """Create a visual compression bar."""
    filled = int(ratio * width)
    empty = width - filled
    bar = Text()
    bar.append("█" * filled, style="bold green")
    bar.append("░" * empty, style="dim")
    bar.append(f" {ratio * 100:.1f}%", style="bold cyan")
    return bar


class Dashboard:
    """
    Real-time terminal dashboard for Token-Guard metrics.

    Runs a Rich Live display in a background thread, updating
    every second with the latest proxy metrics.
    """

    def __init__(self, metrics: SessionMetrics, host: str = "127.0.0.1", port: int = 8000):
        self.metrics = metrics
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._console = Console()
        self._live: Optional[Live] = None

    def start(self) -> None:
        """Start the dashboard in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tg-dashboard")
        self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        """Main dashboard loop."""
        try:
            with Live(
                self._build_display(),
                console=self._console,
                refresh_per_second=2,
                screen=False,
            ) as live:
                self._live = live
                while not self._stop_event.is_set():
                    live.update(self._build_display())
                    time.sleep(0.5)
        except Exception:
            # Dashboard is non-critical — don't crash the server
            pass

    def _build_display(self) -> Panel:
        """Build the full dashboard display."""
        sections = []

        # ── Status Header ──
        status_line = Text()
        status_line.append("  Status: ", style="dim")
        status_line.append("🟢 ", style="bold green")
        status_line.append("Proxying", style="bold green")
        status_line.append(f"  →  ", style="dim")
        status_line.append(f"http://{self.host}:{self.port}", style="bold cyan underline")
        sections.append(status_line)

        # ── Uptime & Requests ──
        info_line = Text()
        info_line.append("  Uptime: ", style="dim")
        info_line.append(_format_uptime(self.metrics.uptime_seconds), style="bold white")
        info_line.append("  │  ", style="dim")
        info_line.append("Requests: ", style="dim")
        info_line.append(str(self.metrics.total_requests), style="bold white")
        sections.append(info_line)

        sections.append(Text(""))  # spacer

        # ── Latest Request Details ──
        latest = self.metrics.latest_request
        if latest and latest.file_stats:
            file_table = Table(
                box=box.SIMPLE_HEAVY,
                show_header=True,
                header_style="bold bright_white",
                padding=(0, 1),
                expand=True,
            )
            file_table.add_column("", width=3, justify="center")
            file_table.add_column("File", style="bold", min_width=20)
            file_table.add_column("Status", min_width=14)
            file_table.add_column("Tokens", justify="right", min_width=12)

            for fstat in latest.file_stats:
                if fstat.was_skeletonized:
                    icon = "🟡"
                    status = Text("Skeletonized", style="yellow")
                    tokens = Text(f"saved {_format_tokens(fstat.tokens_saved)}", style="bold yellow")
                else:
                    icon = "🟢"
                    status = Text("Intact", style="green")
                    tokens = Text(f"kept {_format_tokens(fstat.original_tokens)}", style="dim green")

                file_table.add_row(icon, fstat.filename, status, tokens)

            req_panel = Panel(
                file_table,
                title="[bold]Latest Request[/bold]",
                border_style="bright_black",
                padding=(0, 0),
            )
            sections.append(req_panel)
        elif self.metrics.total_requests == 0:
            waiting = Text("  ⏳ Waiting for first request...", style="dim italic")
            sections.append(waiting)

        sections.append(Text(""))  # spacer

        # ── Session Summary ──
        summary = Text()
        summary.append("  💰 Session: ", style="dim")
        summary.append(f"Saved {_format_tokens(self.metrics.tokens_saved)} tokens", style="bold green")
        summary.append(f" ({_format_money(self.metrics.cost_saved)})", style="bold yellow")
        summary.append("  │  ", style="dim")
        summary.append("Compression: ", style="dim")
        sections.append(summary)

        # Compression bar
        if self.metrics.total_requests > 0:
            bar_line = Text("  ")
            bar_line.append_text(_compression_bar(self.metrics.compression_ratio))
            sections.append(bar_line)

        # ── Build the outer panel ──
        display = Panel(
            Group(*sections),
            title="[bold bright_white]🚀 Token-Guard Active[/bold bright_white]",
            subtitle="[dim]Ctrl+C to stop[/dim]",
            border_style="bright_cyan",
            padding=(1, 2),
            expand=True,
        )

        return display

    def print_startup_banner(self) -> None:
        """Print a one-time startup banner."""
        self._console.print()
        banner = Panel(
            Group(
                Text("🚀 Token-Guard v0.1.0", style="bold bright_cyan", justify="center"),
                Text("Smart Context-Aware Token Compactor", style="dim italic", justify="center"),
                Text("", justify="center"),
                Text(
                    f"Proxy listening on http://{self.host}:{self.port}",
                    style="bold white",
                    justify="center",
                ),
                Text(
                    "Point your coding agent's API base URL here",
                    style="dim",
                    justify="center",
                ),
            ),
            border_style="bright_cyan",
            padding=(1, 4),
        )
        self._console.print(banner)
        self._console.print()
