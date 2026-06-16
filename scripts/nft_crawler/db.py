"""SQLite persistence — crawl state, frontier queue, fetch audit log.

Google's crawler keeps URL seen-sets, link graphs, and per-URL fetch state in
Bigtable. We use SQLite because:
  - single-machine crawler fits our POC scale (millions of tokens, not billions)
  - ACID transactions for idempotent re-crawl
  - zero ops overhead vs running Postgres
  - easy to inspect with `sqlite3` CLI during development

Schema mirrors a simplified version of a web crawler:
  collections  ~ site registry (one row per NFT contract)
  tokens       ~ pages (one row per token, with content hashes)
  frontier     ~ URL frontier priority queue
  fetch_log    ~ HTTP audit trail for debugging / rate-limit tuning
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .models import (
    Chain,
    Collection,
    MetadataSource,
    QueuePriority,
    TokenRecord,
    TokenStatus,
)


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS collections (
    slug            TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    chain           TEXT NOT NULL,
    contract        TEXT NOT NULL,
    supply          INTEGER,
    metadata_source TEXT NOT NULL,
    ipfs_cid        TEXT,
    hf_repo         TEXT,
    opensea_slug    TEXT,
    extra           TEXT DEFAULT '{}',
    discovered_at   REAL NOT NULL,
    last_crawled_at REAL,
    tokens_done     INTEGER DEFAULT 0,
    tokens_failed   INTEGER DEFAULT 0,
    UNIQUE(chain, contract)
);

CREATE TABLE IF NOT EXISTS tokens (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chain           TEXT NOT NULL,
    contract        TEXT NOT NULL,
    token_id        INTEGER NOT NULL,
    collection_slug TEXT NOT NULL REFERENCES collections(slug),
    status          TEXT NOT NULL DEFAULT 'pending',
    name            TEXT,
    description     TEXT,
    image_uri       TEXT,
    animation_uri   TEXT,
    metadata_uri    TEXT,
    traits          TEXT DEFAULT '{}',
    image_path      TEXT,
    metadata_path   TEXT,
    image_sha256    TEXT,
    error           TEXT,
    attempts        INTEGER DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    UNIQUE(chain, contract, token_id)
);

CREATE INDEX IF NOT EXISTS idx_tokens_status ON tokens(status);
CREATE INDEX IF NOT EXISTS idx_tokens_collection ON tokens(collection_slug);
CREATE INDEX IF NOT EXISTS idx_tokens_contract ON tokens(chain, contract);

CREATE TABLE IF NOT EXISTS frontier (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chain           TEXT NOT NULL,
    contract        TEXT NOT NULL,
    token_id        INTEGER NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 1,
    stage           TEXT NOT NULL,  -- 'metadata' | 'image'
    scheduled_at    REAL NOT NULL,
    UNIQUE(chain, contract, token_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_frontier_sched ON frontier(priority, scheduled_at);

CREATE TABLE IF NOT EXISTS fetch_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT NOT NULL,
    host            TEXT NOT NULL,
    status_code     INTEGER,
    bytes           INTEGER,
    latency_ms      REAL,
    error           TEXT,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    tokens_metadata INTEGER DEFAULT 0,
    tokens_images   INTEGER DEFAULT 0,
    tokens_failed   INTEGER DEFAULT 0,
    notes           TEXT
);
"""


class CrawlDB:
    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = __import__("threading").RLock()

    def connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self.path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.executescript(SCHEMA)
            return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self.connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── collections ──────────────────────────────────────────────────────

    def upsert_collection(self, col: Collection) -> None:
        now = time.time()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO collections
                    (slug, name, chain, contract, supply, metadata_source,
                     ipfs_cid, hf_repo, opensea_slug, extra, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    name=excluded.name,
                    supply=COALESCE(excluded.supply, collections.supply),
                    ipfs_cid=COALESCE(excluded.ipfs_cid, collections.ipfs_cid),
                    hf_repo=COALESCE(excluded.hf_repo, collections.hf_repo),
                    opensea_slug=COALESCE(excluded.opensea_slug, collections.opensea_slug),
                    extra=excluded.extra
                """,
                (
                    col.slug,
                    col.name,
                    col.chain.value,
                    col.contract.lower(),
                    col.supply,
                    col.metadata_source.value,
                    col.ipfs_cid,
                    col.hf_repo,
                    col.opensea_slug,
                    json.dumps(col.extra),
                    now,
                ),
            )

    def get_collection(self, slug: str) -> Collection | None:
        row = self.connect().execute(
            "SELECT * FROM collections WHERE slug=?", (slug,)
        ).fetchone()
        return _row_to_collection(row) if row else None

    def list_collections(self) -> list[Collection]:
        rows = self.connect().execute(
            "SELECT * FROM collections ORDER BY discovered_at"
        ).fetchall()
        return [_row_to_collection(r) for r in rows]

    # ── tokens ───────────────────────────────────────────────────────────

    def upsert_tokens(
        self,
        chain: Chain,
        contract: str,
        collection_slug: str,
        token_ids: list[int],
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> int:
        """Insert tokens + enqueue metadata fetch. Returns count of new tokens."""
        now = time.time()
        contract = contract.lower()
        new = 0
        with self.transaction() as conn:
            for tid in token_ids:
                cur = conn.execute(
                    """
                    INSERT INTO tokens
                        (chain, contract, token_id, collection_slug, status,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    ON CONFLICT(chain, contract, token_id) DO NOTHING
                    """,
                    (chain.value, contract, tid, collection_slug, now, now),
                )
                if cur.rowcount:
                    new += 1
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO frontier
                            (chain, contract, token_id, priority, stage, scheduled_at)
                        VALUES (?, ?, ?, ?, 'metadata', ?)
                        """,
                        (chain.value, contract, tid, priority.value, now),
                    )
        return new

    def get_token(self, chain: Chain, contract: str, token_id: int) -> TokenRecord | None:
        row = self.connect().execute(
            """
            SELECT * FROM tokens
            WHERE chain=? AND contract=? AND token_id=?
            """,
            (chain.value, contract.lower(), token_id),
        ).fetchone()
        return _row_to_token(row) if row else None

    def update_token(self, rec: TokenRecord) -> None:
        now = time.time()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE tokens SET
                    status=?, name=?, description=?,
                    image_uri=?, animation_uri=?, metadata_uri=?,
                    traits=?, image_path=?, metadata_path=?,
                    image_sha256=?, error=?, attempts=?, updated_at=?
                WHERE chain=? AND contract=? AND token_id=?
                """,
                (
                    rec.status.value,
                    rec.name,
                    rec.description,
                    rec.image_uri,
                    rec.animation_uri,
                    rec.metadata_uri,
                    json.dumps(rec.traits),
                    rec.image_path,
                    rec.metadata_path,
                    rec.image_sha256,
                    rec.error,
                    rec.attempts,
                    now,
                    rec.chain.value,
                    rec.contract.lower(),
                    rec.token_id,
                ),
            )

    def count_tokens_by_status(self, collection_slug: str | None = None) -> dict[str, int]:
        if collection_slug:
            rows = self.connect().execute(
                "SELECT status, COUNT(*) c FROM tokens WHERE collection_slug=? GROUP BY status",
                (collection_slug,),
            ).fetchall()
        else:
            rows = self.connect().execute(
                "SELECT status, COUNT(*) c FROM tokens GROUP BY status"
            ).fetchall()
        return {r["status"]: r["c"] for r in rows}

    # ── frontier ─────────────────────────────────────────────────────────

    def pop_frontier(self, stage: str, limit: int = 100) -> list[tuple[str, str, int]]:
        """Atomically claim up to `limit` frontier items for a stage."""
        now = time.time()
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, chain, contract, token_id FROM frontier
                WHERE stage=? AND scheduled_at <= ?
                ORDER BY priority ASC, scheduled_at ASC
                LIMIT ?
                """,
                (stage, now, limit),
            ).fetchall()
            if not rows:
                return []
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM frontier WHERE id IN ({placeholders})", ids)
            return [(r["chain"], r["contract"], r["token_id"]) for r in rows]

    def enqueue(
        self,
        chain: Chain,
        contract: str,
        token_id: int,
        stage: str,
        priority: QueuePriority = QueuePriority.LOW,
        delay_s: float = 0,
    ) -> None:
        scheduled = time.time() + delay_s
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO frontier (chain, contract, token_id, priority, stage, scheduled_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chain, contract, token_id, stage) DO UPDATE SET
                    priority=MIN(frontier.priority, excluded.priority),
                    scheduled_at=MIN(frontier.scheduled_at, excluded.scheduled_at)
                """,
                (chain.value, contract.lower(), token_id, priority.value, stage, scheduled),
            )

    def frontier_depth(self) -> dict[str, int]:
        rows = self.connect().execute(
            "SELECT stage, COUNT(*) c FROM frontier GROUP BY stage"
        ).fetchall()
        return {r["stage"]: r["c"] for r in rows}

    # ── fetch log ────────────────────────────────────────────────────────

    def log_fetch(
        self,
        url: str,
        host: str,
        status_code: int | None,
        nbytes: int,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        with self._lock:
            conn = self.connect()
            conn.execute(
                """
                INSERT INTO fetch_log (url, host, status_code, bytes, latency_ms, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, host, status_code, nbytes, latency_ms, error, time.time()),
            )
            conn.commit()

    # ── crawl runs ───────────────────────────────────────────────────────

    def start_run(self) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO crawl_runs (started_at) VALUES (?)", (time.time(),)
            )
            return cur.lastrowid  # type: ignore[return-value]

    def finish_run(
        self,
        run_id: int,
        metadata: int,
        images: int,
        failed: int,
        notes: str = "",
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE crawl_runs SET
                    ended_at=?, tokens_metadata=?, tokens_images=?,
                    tokens_failed=?, notes=?
                WHERE id=?
                """,
                (time.time(), metadata, images, failed, notes, run_id),
            )

    def repair_frontier(self, collection_slug: str | None = None) -> int:
        """Re-enqueue tokens stuck without frontier entries."""
        now = time.time()
        n = 0
        with self.transaction() as conn:
            q = "SELECT chain, contract, token_id, status FROM tokens WHERE status IN ('pending', 'metadata_fetched')"
            params: tuple = ()
            if collection_slug:
                q += " AND collection_slug=?"
                params = (collection_slug,)
            rows = conn.execute(q, params).fetchall()
            for row in rows:
                stage = "metadata" if row["status"] == "pending" else "image"
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO frontier
                        (chain, contract, token_id, priority, stage, scheduled_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (row["chain"], row["contract"], row["token_id"], QueuePriority.LOW.value, stage, now),
                )
                n += cur.rowcount
        return n

    def stats(self) -> dict[str, Any]:
        conn = self.connect()
        col_count = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
        tok_count = conn.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
        frontier = self.frontier_depth()
        by_status = self.count_tokens_by_status()
        last_run = conn.execute(
            "SELECT * FROM crawl_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "collections": col_count,
            "tokens": tok_count,
            "tokens_by_status": by_status,
            "frontier": frontier,
            "last_run": dict(last_run) if last_run else None,
        }


def _row_to_collection(row: sqlite3.Row) -> Collection:
    return Collection(
        slug=row["slug"],
        name=row["name"],
        chain=Chain(row["chain"]),
        contract=row["contract"],
        supply=row["supply"],
        metadata_source=MetadataSource(row["metadata_source"]),
        ipfs_cid=row["ipfs_cid"],
        hf_repo=row["hf_repo"],
        opensea_slug=row["opensea_slug"],
        extra=json.loads(row["extra"] or "{}"),
    )


def _row_to_token(row: sqlite3.Row) -> TokenRecord:
    return TokenRecord(
        chain=Chain(row["chain"]),
        contract=row["contract"],
        token_id=row["token_id"],
        collection_slug=row["collection_slug"],
        status=TokenStatus(row["status"]),
        name=row["name"],
        description=row["description"],
        image_uri=row["image_uri"],
        animation_uri=row["animation_uri"],
        metadata_uri=row["metadata_uri"],
        traits=json.loads(row["traits"] or "{}"),
        image_path=row["image_path"],
        metadata_path=row["metadata_path"],
        image_sha256=row["image_sha256"],
        error=row["error"],
        attempts=row["attempts"],
    )
