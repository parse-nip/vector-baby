"""Crawl scheduler — the main loop tying frontier, fetchers, and storage together.

Architecture mirrors Google's fetch cycle:

  1. Scheduler pops batch from frontier (priority queue)
  2. Workers fetch metadata / images in parallel
  3. Parser extracts links (image_uri) → enqueues image stage
  4. Storage writes raw bytes + updates token state
  5. Failed items re-enqueued with backoff

Two-stage pipeline (metadata → image) because:
  - Metadata JSON is tiny (~2 KB); images are large (~100 KB–2 MB)
  - Different rate limits apply (IPFS gateways vs image CDNs)
  - We can re-fetch images without re-parsing metadata
  - Matches Google's separate HTML fetch vs resource fetch
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import CrawlerConfig
from .db import CrawlDB
from .discovery import enumerate_token_ids
from .fetcher import Fetcher
from .models import Chain, Collection, QueuePriority, TokenRecord, TokenStatus
from .parser import extract_fields, parse_metadata_json
from .providers import fetch_token_image, get_provider
from .resolver import image_extension_from_content_type


class Crawler:
    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.raw_dir, exist_ok=True)
        os.makedirs(config.metadata_dir, exist_ok=True)
        os.makedirs(config.images_dir, exist_ok=True)
        self.db = CrawlDB(config.db_path)
        self.fetcher = Fetcher(config, self.db)

    def seed_collection(
        self,
        collection: Collection,
        limit: int | None = None,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> int:
        """Register collection and enqueue all token IDs."""
        self.db.upsert_collection(collection)
        token_ids = enumerate_token_ids(collection, limit=limit)
        return self.db.upsert_tokens(
            collection.chain,
            collection.contract,
            collection.slug,
            token_ids,
            priority=priority,
        )

    def _token_paths(self, chain: Chain, contract: str, token_id: int) -> tuple[str, str]:
        base = f"{chain.value}/{contract.lower()}/{token_id}"
        meta_path = os.path.join(self.config.metadata_dir, f"{base}.json")
        return meta_path, base

    def _process_metadata(self, chain_s: str, contract: str, token_id: int) -> bool:
        chain = Chain(chain_s)
        rec = self.db.get_token(chain, contract, token_id)
        if not rec:
            return False

        col = self.db.get_collection(rec.collection_slug)
        if not col:
            rec.status = TokenStatus.FAILED
            rec.error = "collection not found"
            rec.attempts += 1
            self.db.update_token(rec)
            return False

        provider = get_provider(col.metadata_source)
        raw, uri, err = provider.fetch_metadata(col, token_id, self.fetcher, self.config)

        if raw is None:
            rec.attempts += 1
            rec.error = err or "metadata fetch failed"
            if rec.attempts >= self.config.max_retries:
                rec.status = TokenStatus.FAILED
                self.db.update_token(rec)
                return False
            # Exponential backoff re-queue
            delay = min(300, 2 ** rec.attempts)
            self.db.enqueue(chain, contract, token_id, "metadata", QueuePriority.LOW, delay)
            self.db.update_token(rec)
            return False

        try:
            data = parse_metadata_json(raw)
            fields = extract_fields(data)
        except Exception as e:
            rec.attempts += 1
            rec.error = f"parse error: {e}"
            if rec.attempts >= self.config.max_retries:
                rec.status = TokenStatus.FAILED
            self.db.update_token(rec)
            return False

        meta_path, _ = self._token_paths(chain, contract, token_id)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(data, f)

        rec.name = fields.get("name")
        rec.description = fields.get("description")
        rec.image_uri = fields.get("image_uri")
        rec.animation_uri = fields.get("animation_uri")
        rec.metadata_uri = uri
        rec.metadata_path = meta_path
        rec.traits = fields.get("traits", {})
        rec.status = TokenStatus.METADATA_FETCHED
        rec.error = None
        self.db.update_token(rec)

        if rec.image_uri:
            self.db.enqueue(chain, contract, token_id, "image", QueuePriority.NORMAL)
        else:
            rec.status = TokenStatus.SKIPPED
            rec.error = "no image_uri in metadata"
            self.db.update_token(rec)
        return True

    def _process_image(self, chain_s: str, contract: str, token_id: int) -> bool:
        chain = Chain(chain_s)
        rec = self.db.get_token(chain, contract, token_id)
        if not rec or not rec.image_uri:
            return False

        result = fetch_token_image(rec.image_uri, self.fetcher, self.config)
        if not result.ok or not result.data:
            rec.attempts += 1
            rec.error = result.error or "image fetch failed"
            if rec.attempts >= self.config.max_retries:
                rec.status = TokenStatus.FAILED
                self.db.update_token(rec)
                return False
            delay = min(300, 2 ** rec.attempts)
            self.db.enqueue(chain, contract, token_id, "image", QueuePriority.LOW, delay)
            self.db.update_token(rec)
            return False

        ext = image_extension_from_content_type(result.content_type or "", result.data)
        _, base = self._token_paths(chain, contract, token_id)
        img_path = os.path.join(self.config.images_dir, f"{base}.{ext}")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(result.data)

        rec.image_path = img_path
        rec.image_sha256 = Fetcher.sha256(result.data)
        rec.status = TokenStatus.DONE
        rec.error = None
        self.db.update_token(rec)
        return True

    def run(
        self,
        max_metadata: int = 500,
        max_images: int = 500,
        collection_slug: str | None = None,
    ) -> dict:
        """Run one crawl cycle. Returns stats dict."""
        run_id = self.db.start_run()
        t0 = time.time()
        meta_ok = img_ok = failed = 0

        # Metadata stage
        meta_batch = self.db.pop_frontier("metadata", limit=max_metadata)
        if meta_batch:
            print(f"  metadata frontier: {len(meta_batch)} items", flush=True)
            with ThreadPoolExecutor(max_workers=self.config.metadata_workers) as ex:
                futs = {
                    ex.submit(self._process_metadata, c, a, t): (c, a, t)
                    for c, a, t in meta_batch
                }
                for fut in as_completed(futs):
                    try:
                        if fut.result():
                            meta_ok += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1

        # Image stage
        img_batch = self.db.pop_frontier("image", limit=max_images)
        if img_batch:
            print(f"  image frontier: {len(img_batch)} items", flush=True)
            with ThreadPoolExecutor(max_workers=self.config.image_workers) as ex:
                futs = {
                    ex.submit(self._process_image, c, a, t): (c, a, t)
                    for c, a, t in img_batch
                }
                for fut in as_completed(futs):
                    try:
                        if fut.result():
                            img_ok += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1

        elapsed = time.time() - t0
        self.db.finish_run(run_id, meta_ok, img_ok, failed, f"elapsed={elapsed:.1f}s")
        return {
            "metadata_fetched": meta_ok,
            "images_fetched": img_ok,
            "failed": failed,
            "elapsed_s": elapsed,
            "frontier": self.db.frontier_depth(),
        }

    def run_until_empty(
        self,
        batch_metadata: int = 200,
        batch_images: int = 200,
        max_rounds: int = 10000,
    ) -> None:
        """Keep crawling until frontier is empty or max_rounds hit."""
        for rnd in range(max_rounds):
            frontier = self.db.frontier_depth()
            if not frontier:
                print("frontier empty — crawl complete", flush=True)
                break
            print(f"round {rnd + 1}: frontier={frontier}", flush=True)
            stats = self.run(max_metadata=batch_metadata, max_images=batch_images)
            print(f"  -> {stats}", flush=True)
            if stats["metadata_fetched"] == 0 and stats["images_fetched"] == 0:
                # Nothing processed — likely all scheduled in future (backoff)
                pending = sum(frontier.values())
                if pending > 0:
                    print(f"  waiting for {pending} backoff items...", flush=True)
                    time.sleep(5)
                else:
                    break

    def close(self) -> None:
        self.db.close()
