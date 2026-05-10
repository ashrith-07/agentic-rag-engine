"""
Scheduled workflow ping for Qdrant.

Qdrant Cloud (*.cloud.qdrant.io) requires an API key for programmatic access — add
repository secret QDRANT_API_KEY. Self-hosted or HTTP URLs without auth may omit it.

For a bare hostname with no key: HTTPS for remote hosts, HTTP host:port for loopback only.
"""

import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


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

    if not api_key and "cloud.qdrant.io" in host.lower():
        print(
            "error: Qdrant Cloud requires an API key for REST access from CI.\n"
            "In Qdrant Cloud: open your cluster → Access / API keys → create a key.\n"
            "Add GitHub repository secret QDRANT_API_KEY with that value.",
            file=sys.stderr,
        )
        sys.exit(1)

    if host.startswith("http"):
        if api_key:
            return QdrantClient(url=host, api_key=api_key, timeout=30)
        return QdrantClient(url=host, timeout=30)

    if api_key:
        return QdrantClient(url=f"https://{host}", api_key=api_key, timeout=30)

    if _bare_host_is_local(host):
        return QdrantClient(host=host, port=port, timeout=30)

    # Self-hosted or other HTTPS endpoint without auth (unusual for production)
    return QdrantClient(url=f"https://{host}", timeout=30)


def main() -> None:
    client = make_client()
    try:
        collections = client.get_collections()
    except UnexpectedResponse as exc:
        if exc.status_code == 403:
            print(
                "error: Qdrant returned 403 Forbidden.\n"
                "If this cluster uses API-key auth, set repository secret QDRANT_API_KEY.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    n = len(collections.collections)
    print(f"✓ Qdrant keep-alive ping — {n} collections active")


if __name__ == "__main__":
    main()
