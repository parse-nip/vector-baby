"""HTTP / IPFS / Arweave fetcher with gateway rotation and retries.

Design: separate transport from business logic. The fetcher only knows how to
retrieve bytes from a URI; the resolver turns tokenURI strings into concrete
URLs; the parser extracts fields from JSON.

Gateway rotation mirrors fetch_traits.py but generalizes to arbitrary CIDs
and paths. On failure we try the next gateway before giving up.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from .models import FetchResult
from .politeness import PolitenessManager

if TYPE_CHECKING:
    from .config import CrawlerConfig
    from .db import CrawlDB


class Fetcher:
    def __init__(self, config: CrawlerConfig, db: CrawlDB | None = None) -> None:
        self.config = config
        self.db = db
        self.politeness = PolitenessManager(config.min_delay_per_host)

    def fetch(self, url: str) -> FetchResult:
        """Fetch a single HTTP(S) URL with politeness + logging."""
        self.politeness.wait(url)
        t0 = time.monotonic()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self.config.user_agent},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                data = resp.read()
                ct = resp.headers.get("Content-Type", "")
                latency = (time.monotonic() - t0) * 1000
                if self.db:
                    self.db.log_fetch(
                        url,
                        PolitenessManager.host_of(url),
                        resp.status,
                        len(data),
                        latency,
                    )
                return FetchResult(
                    ok=True,
                    data=data,
                    content_type=ct,
                    final_url=url,
                    status_code=resp.status,
                )
        except urllib.error.HTTPError as e:
            latency = (time.monotonic() - t0) * 1000
            if self.db:
                self.db.log_fetch(
                    url,
                    PolitenessManager.host_of(url),
                    e.code,
                    0,
                    latency,
                    str(e),
                )
            return FetchResult(ok=False, error=str(e), status_code=e.code, final_url=url)
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            if self.db:
                self.db.log_fetch(
                    url,
                    PolitenessManager.host_of(url),
                    None,
                    0,
                    latency,
                    str(e),
                )
            return FetchResult(ok=False, error=str(e), final_url=url)

    def fetch_with_gateways(self, path: str, gateways: list[str]) -> FetchResult:
        """Try each gateway template until one succeeds. `path` is the IPFS/Arweave path."""
        last_err: str | None = None
        for tmpl in gateways:
            url = tmpl.format(path=path)
            result = self.fetch(url)
            if result.ok and result.data:
                return result
            last_err = result.error
        return FetchResult(ok=False, error=last_err or "all gateways failed")

    def fetch_resolved(self, resolved_urls: list[str]) -> FetchResult:
        """Try a list of already-resolved HTTP URLs in order."""
        last_err: str | None = None
        for url in resolved_urls:
            result = self.fetch(url)
            if result.ok and result.data:
                return result
            last_err = result.error
        return FetchResult(ok=False, error=last_err or "all URLs failed")

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
