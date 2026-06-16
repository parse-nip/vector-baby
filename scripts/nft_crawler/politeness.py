"""Per-host rate limiting — crawler politeness layer.

Googlebot respects robots.txt and crawl-delay. NFT metadata lives on
heterogeneous hosts (ipfs.io, Pinata, Alchemy, custom CDNs). We enforce a
minimum inter-request delay per host to avoid getting blocked and to be a
good citizen on public gateways.
"""

from __future__ import annotations

import threading
import time
from urllib.parse import urlparse


class PolitenessManager:
    def __init__(self, min_delay: float = 0.25) -> None:
        self.min_delay = min_delay
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        host = urlparse(url).netloc or url
        with self._lock:
            last = self._last_request.get(host, 0.0)
            now = time.monotonic()
            sleep_for = self.min_delay - (now - last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_request[host] = time.monotonic()

    @staticmethod
    def host_of(url: str) -> str:
        return urlparse(url).netloc or url
