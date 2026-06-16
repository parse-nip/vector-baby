"""URI resolution — turn ipfs://, ar://, data: URIs into fetchable HTTP URLs.

ERC-721 tokenURI returns strings like:
  ipfs://Qm.../123
  https://api.example.com/metadata/123
  ar://txid

This module normalizes them into ordered lists of HTTP URLs to try.
"""

from __future__ import annotations

import base64
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import CrawlerConfig

_IPFS_PREFIXES = ("ipfs://", "ipns://")
_AR_PREFIX = "ar://"
_DATA_PREFIX = "data:"


def resolve_uri(uri: str, config: CrawlerConfig) -> list[str]:
    """Return HTTP URLs to try, in priority order."""
    uri = uri.strip()

    if uri.startswith("https://") or uri.startswith("http://"):
        return [uri]

    if uri.startswith(_AR_PREFIX):
        txid = uri[len(_AR_PREFIX) :]
        return [g.format(path=txid) for g in config.arweave_gateways]

    for prefix in _IPFS_PREFIXES:
        if uri.startswith(prefix):
            path = uri[len(prefix) :]
            # CID/path may be Qm.../123 or bafy.../123
            return [g.format(path=path) for g in config.ipfs_gateways]

    # Some contracts return bare CIDs
    if re.match(r"^Qm[a-zA-Z0-9]{44}", uri) or uri.startswith("baf"):
        return [g.format(path=uri) for g in config.ipfs_gateways]

    return [uri]


def parse_data_uri(uri: str) -> bytes | None:
    """Decode data:application/json;base64,... inline metadata."""
    if not uri.startswith(_DATA_PREFIX):
        return None
    try:
        header, payload = uri.split(",", 1)
        if ";base64" in header:
            return base64.b64decode(payload)
        return payload.encode("utf-8")
    except Exception:
        return None


def metadata_uri_for_token(collection, token_id: int) -> str | None:
    """Build the metadata URI for a token given collection config."""
    from .models import MetadataSource

    if collection.metadata_source == MetadataSource.IPFS_PATTERN:
        if not collection.ipfs_cid:
            return None
        return f"ipfs://{collection.ipfs_cid}/{token_id}"

    if collection.metadata_source == MetadataSource.TOKEN_URI:
        # Resolved at crawl time via on-chain call or API
        return None

    return None


def image_extension_from_content_type(ct: str, data: bytes) -> str:
    ct = (ct or "").lower()
    if "png" in ct or data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if "gif" in ct:
        return "gif"
    if "webp" in ct or data[:4] == b"RIFF":
        return "webp"
    if "svg" in ct or data.lstrip()[:5] == b"<svg ":
        return "svg"
    return "jpg"
