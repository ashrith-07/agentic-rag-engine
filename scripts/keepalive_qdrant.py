"""
Scheduled workflow ping for Qdrant.

Connection rules mirror the app for keyed / explicit URLs; for a bare hostname with no
API key we use HTTPS to match typical remote endpoints (original keepalive behavior).
Plain HTTP host:port is only used for obvious local hosts (localhost / loopback).
"""

import os
import sys

from qdrant_client import QdrantClient


def _bare_host_is_local(host: str) -> bool:
    h = host.lower()
    return h in ("localhost", "127.0.0.1", "::1")


def make_client() -> QdrantClient:
    host = os.environ.get("QDRANT_HOST", "").strip()
    if not host:
        print(
            "error: QDRANT_HOST is unset or empty. "
            "Add repository secret QDRANT_HOST (hostname or https://… URL).",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key_raw = os.environ.get("QDRANT_API_KEY")
    api_key = api_key_raw.strip() if api_key_raw else None
    if api_key == "":
        api_key = None

    port_raw = (os.environ.get("QDRANT_PORT") or "").strip()
    port = int(port_raw) if port_raw else 6333

    if host.startswith("http"):
        if api_key:
            return QdrantClient(url=host, api_key=api_key, timeout=30)
        return QdrantClient(url=host, timeout=30)

    if api_key:
        return QdrantClient(url=f"https://{host}", api_key=api_key, timeout=30)

    if _bare_host_is_local(host):
        return QdrantClient(host=host, port=port, timeout=30)

    # Remote host, no key — HTTPS only (e.g. Qdrant Cloud trial / open ingress)
    return QdrantClient(url=f"https://{host}", timeout=30)


def main() -> None:
    client = make_client()
    collections = client.get_collections()
    n = len(collections.collections)
    print(f"✓ Qdrant keep-alive ping — {n} collections active")


if __name__ == "__main__":
    main()
