"""
Direct DNS resolver for TokenTamer upstream forwarding.

When /etc/hosts points api.openai.com and api.anthropic.com to 127.0.0.1,
we need to bypass that mapping to reach the real upstream servers.

Uses subprocess `dig` or `nslookup` to query public DNS directly,
plus a socket.getaddrinfo monkey-patch for clean httpx integration.
"""

from __future__ import annotations

import socket
import subprocess
from contextlib import contextmanager
from typing import Dict, Optional

# Domains we intercept — these must bypass /etc/hosts when forwarding upstream
INTERCEPT_DOMAINS = {"api.openai.com", "api.anthropic.com"}

# Well-known fallback IPs (last resort if DNS fails)
FALLBACK_IPS: Dict[str, str] = {
    "api.openai.com": "104.18.7.192",
    "api.anthropic.com": "104.18.20.212",
}

_DNS_CACHE: Dict[str, str] = {}
_original_getaddrinfo = socket.getaddrinfo


def resolve_host(host: str) -> str:
    """
    Resolve a hostname directly via public DNS, bypassing /etc/hosts.
    Falls back to hardcoded IPs if all else fails.
    """
    if host in _DNS_CACHE:
        return _DNS_CACHE[host]

    ip = _resolve_via_dig(host)
    if not ip:
        ip = _resolve_via_nslookup(host)
    if not ip:
        ip = _resolve_via_socket(host)
    if not ip:
        ip = FALLBACK_IPS.get(host, host)

    _DNS_CACHE[host] = ip
    return ip


def _resolve_via_dig(host: str) -> Optional[str]:
    """Use dig with a public DNS server."""
    try:
        result = subprocess.run(
            ["dig", "+short", "+time=2", "@8.8.8.8", host, "A"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line and not line.startswith(";") and "." in line:
                return line
    except Exception:
        pass
    return None


def _resolve_via_nslookup(host: str) -> Optional[str]:
    """Use nslookup with a public DNS server."""
    try:
        result = subprocess.run(
            ["nslookup", "-timeout=2", host, "8.8.8.8"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        in_answer = False
        for line in result.stdout.splitlines():
            if "Name:" in line:
                in_answer = True
                continue
            if in_answer and "Address:" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    return parts[1].strip()
    except Exception:
        pass
    return None


def _resolve_via_socket(host: str) -> Optional[str]:
    """Standard socket resolution (will use /etc/hosts, so least preferred)."""
    try:
        return socket.gethostbyname(host)
    except Exception:
        return None


# ── Socket monkey-patch for httpx ──

def _patched_getaddrinfo(host, port, *args, **kwargs):
    if isinstance(host, str) and host in INTERCEPT_DOMAINS:
        ip = resolve_host(host)
        return _original_getaddrinfo(ip, port, *args, **kwargs)
    return _original_getaddrinfo(host, port, *args, **kwargs)


@contextmanager
def bypass_hosts():
    """
    Context manager that patches socket.getaddrinfo so httpx/requests
    resolve intercept domains via public DNS instead of /etc/hosts.
    """
    socket.getaddrinfo = _patched_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = _original_getaddrinfo
