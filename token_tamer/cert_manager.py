"""
SSL Certificate Manager for TokenTamer HTTPS interception.

Generates a local CA and server certificates with SANs for:
  - api.openai.com
  - api.anthropic.com

Allows TokenTamer to transparently intercept HTTPS traffic from
hardcoded clients like Claude Code and Codex CLI.
"""

from __future__ import annotations

import datetime
import os
import subprocess
from pathlib import Path
from typing import Optional


# Domains we want to intercept
INTERCEPT_DOMAINS = ["api.openai.com", "api.anthropic.com"]

DEFAULT_CERT_DIR = Path.home() / ".config" / "token-tamer" / "certs"


class CertManager:
    """Manages local CA and per-domain certificates."""

    def __init__(self, cert_dir: Optional[Path] = None):
        self.cert_dir = cert_dir or DEFAULT_CERT_DIR
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        self.ca_key = self.cert_dir / "ca-key.pem"
        self.ca_cert = self.cert_dir / "ca-cert.pem"
        self.server_key = self.cert_dir / "server-key.pem"
        self.server_cert = self.cert_dir / "server-cert.pem"

    def ensure_ca(self) -> Path:
        """Generate a local root CA if it doesn't exist."""
        if self.ca_cert.exists() and self.ca_key.exists():
            return self.ca_cert

        # Generate CA private key
        subprocess.run(
            ["openssl", "genrsa", "-out", str(self.ca_key), "2048"],
            check=True, capture_output=True,
        )
        # Generate self-signed CA certificate
        subprocess.run(
            [
                "openssl", "req", "-x509", "-new", "-nodes",
                "-key", str(self.ca_key),
                "-sha256", "-days", "365",
                "-out", str(self.ca_cert),
                "-subj", "/CN=TokenTamer Local CA/O=TokenTamer",
            ],
            check=True, capture_output=True,
        )
        return self.ca_cert

    def ensure_server_cert(self) -> tuple[Path, Path]:
        """Generate a server certificate with SANs for all intercept domains."""
        if self.server_cert.exists() and self.server_key.exists():
            return self.server_cert, self.server_key

        self.ensure_ca()

        # Generate server private key
        subprocess.run(
            ["openssl", "genrsa", "-out", str(self.server_key), "2048"],
            check=True, capture_output=True,
        )

        # Build SAN config
        san_list = ",".join(f"DNS:{d}" for d in INTERCEPT_DOMAINS)
        ext_conf = self.cert_dir / "ext.cnf"
        ext_conf.write_text(
            f"[v3_ca]\nsubjectAltName={san_list}\nbasicConstraints=CA:FALSE\n"
        )

        # Generate CSR
        csr = self.cert_dir / "server.csr"
        subprocess.run(
            [
                "openssl", "req", "-new",
                "-key", str(self.server_key),
                "-out", str(csr),
                "-subj", f"/CN={INTERCEPT_DOMAINS[0]}",
            ],
            check=True, capture_output=True,
        )

        # Sign with CA
        subprocess.run(
            [
                "openssl", "x509", "-req", "-in", str(csr),
                "-CA", str(self.ca_cert), "-CAkey", str(self.ca_key),
                "-CAcreateserial",
                "-out", str(self.server_cert),
                "-days", "365", "-sha256",
                "-extfile", str(ext_conf), "-extensions", "v3_ca",
            ],
            check=True, capture_output=True,
        )
        return self.server_cert, self.server_key

    def install_ca_macos(self) -> bool:
        """Add the CA certificate to the macOS system trust store."""
        if not self.ca_cert.exists():
            self.ensure_ca()
        try:
            subprocess.run(
                [
                    "sudo", "security", "add-trusted-cert",
                    "-d", "-r", "trustRoot",
                    "-k", "/Library/Keychains/System.keychain",
                    str(self.ca_cert),
                ],
                check=True, capture_output=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install CA (needs sudo): {e}")
            return False

    def uninstall_ca_macos(self) -> bool:
        """Remove the CA certificate from macOS trust store."""
        try:
            subprocess.run(
                [
                    "sudo", "security", "remove-trusted-cert",
                    "-d", str(self.ca_cert),
                ],
                check=False, capture_output=True,
            )
            return True
        except Exception:
            return False

    def print_instructions(self) -> None:
        """Print manual trust-install instructions."""
        print("\n" + "=" * 60)
        print("HTTPS Interception Setup")
        print("=" * 60)
        print(f"\n1. CA certificate: {self.ca_cert}")
        print(f"   Server cert:    {self.server_cert}")
        print(f"   Server key:     {self.server_key}")
        print("\n2. Trust the CA (macOS):")
        print(f"   sudo security add-trusted-cert -d -r trustRoot \\")
        print(f"     -k /Library/Keychains/System.keychain {self.ca_cert}")
        print("\n3. Edit /etc/hosts (needs sudo):")
        print("   sudo nano /etc/hosts")
        for domain in INTERCEPT_DOMAINS:
            print(f"   127.0.0.1 {domain}")
        print("\n4. Run TokenTamer with --ssl")
        print("   token-tamer --ssl --port 443   # or 8443 without sudo")
        print("=" * 60 + "\n")
