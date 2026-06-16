"""Core data types for the NFT crawler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Chain(str, Enum):
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BASE = "base"
    ARBITRUM = "arbitrum"


class TokenStatus(str, Enum):
    PENDING = "pending"
    METADATA_FETCHED = "metadata_fetched"
    IMAGE_FETCHED = "image_fetched"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"  # no image (e.g. burned token)


class QueuePriority(int, Enum):
    HIGH = 0      # newly discovered
    NORMAL = 1    # standard crawl
    LOW = 2       # retry / backfill
    BACKFILL = 3  # historical sweep


class MetadataSource(str, Enum):
    """How we resolve token metadata for a collection."""

    IPFS_PATTERN = "ipfs_pattern"  # base CID + /{token_id}
    TOKEN_URI = "token_uri"        # on-chain ERC-721 tokenURI()
    HUGGINGFACE = "huggingface"    # HF dataset parquet
    HTTP_API = "http_api"          # marketplace / indexer API


@dataclass
class Collection:
    slug: str
    name: str
    chain: Chain
    contract: str  # checksummed or lowercase 0x...
    supply: int | None = None
    metadata_source: MetadataSource = MetadataSource.TOKEN_URI
    # ipfs_pattern: base CID (folder containing 0, 1, 2, ...)
    ipfs_cid: str | None = None
    # huggingface: dataset repo id
    hf_repo: str | None = None
    # optional marketplace slug for discovery expansion
    opensea_slug: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenRecord:
    chain: Chain
    contract: str
    token_id: int
    collection_slug: str
    status: TokenStatus = TokenStatus.PENDING
    name: str | None = None
    description: str | None = None
    image_uri: str | None = None
    animation_uri: str | None = None
    metadata_uri: str | None = None
    traits: dict[str, str] = field(default_factory=dict)
    image_path: str | None = None
    metadata_path: str | None = None
    image_sha256: str | None = None
    error: str | None = None
    attempts: int = 0


@dataclass
class FetchResult:
    ok: bool
    data: bytes | None = None
    content_type: str | None = None
    final_url: str | None = None
    error: str | None = None
    status_code: int | None = None
