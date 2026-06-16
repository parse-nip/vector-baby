"""Crawler configuration — defaults and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_IPFS_GATEWAYS = [
    "https://ipfs.io/ipfs/{path}",
    "https://gateway.pinata.cloud/ipfs/{path}",
    "https://cloudflare-ipfs.com/ipfs/{path}",
    "https://dweb.link/ipfs/{path}",
]

DEFAULT_ARWEAVE_GATEWAYS = [
    "https://arweave.net/{path}",
    "https://ar-io.net/{path}",
]


@dataclass
class CrawlerConfig:
    data_dir: str = "data/crawl"
    db_path: str | None = None  # defaults to {data_dir}/crawl.db

    # concurrency
    metadata_workers: int = 8
    image_workers: int = 12

    # politeness (seconds between requests per host)
    min_delay_per_host: float = 0.25
    max_retries: int = 4
    request_timeout: float = 30.0

    # optional API keys (None = use public gateways only)
    alchemy_api_key: str | None = None
    reservoir_api_key: str | None = None

    ipfs_gateways: list[str] = field(default_factory=lambda: list(DEFAULT_IPFS_GATEWAYS))
    arweave_gateways: list[str] = field(default_factory=lambda: list(DEFAULT_ARWEAVE_GATEWAYS))

    # User-Agent identifies us politely (like Googlebot)
    user_agent: str = "NftCrawler/0.1 (+https://github.com/parse-nip/vector-baby; nft-indexing)"

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = os.path.join(self.data_dir, "crawl.db")
        self.alchemy_api_key = self.alchemy_api_key or os.environ.get("ALCHEMY_API_KEY")
        self.reservoir_api_key = self.reservoir_api_key or os.environ.get("RESERVOIR_API_KEY")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def metadata_dir(self) -> str:
        return os.path.join(self.data_dir, "metadata")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.data_dir, "images")
