# NFT Crawler Architecture

> Google-style discovery and fetch pipeline for NFT metadata and images.
> Feeds the vector-baby semantic search stack ("Exa for NFT images").

---

## 1. The Big Picture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         NFT CRAWLER (Python)                             │
│                                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌────────────┐  │
│  │  Discovery  │──▶│   Frontier   │──▶│   Fetcher   │──▶│  Storage   │  │
│  │  (seeds)    │   │  (priority   │   │  (HTTP/IPFS │   │  (SQLite + │  │
│  │             │   │   queue)     │   │   gateways) │   │   files)   │  │
│  └─────────────┘   └──────────────┘   └─────────────┘   └────────────┘  │
│         │                                    │                  │        │
│         │                                    ▼                  │        │
│         │                             ┌─────────────┐           │        │
│         │                             │   Parser    │           │        │
│         │                             │ (metadata   │           │        │
│         │                             │  JSON)      │           │        │
│         │                             └─────────────┘           │        │
│         │                                                        │        │
│         └────────────────────────────────────────────────────────┘        │
│                                    │                                     │
│                                    ▼                                     │
│                           ┌───────────────┐                              │
│                           │    Export     │                              │
│                           │ manifest.jsonl│                              │
│                           └───────────────┘                              │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING PIPELINE (Python CLIP)                      │
│                                                                          │
│   embed_collection.py  →  embeddings.f32 + meta.json + images/*.jpg     │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    SEARCH ENGINE (Rust vbaby)                            │
│                                                                          │
│   serve-nft  →  exact cosine top-k over CLIP vectors                    │
│   nft_search_app.py  →  text query → CLIP encode → search              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Analogy to Google

| Google Crawler | NFT Crawler |
|----------------|-------------|
| URL frontier | Token frontier `(chain, contract, token_id)` |
| robots.txt / crawl-delay | Per-host rate limiting (`PolitenessManager`) |
| HTML fetch | Metadata JSON fetch |
| Resource fetch (images, CSS) | Image fetch (separate stage) |
| Link extraction | `image_uri` extraction from metadata |
| Page index | SQLite `tokens` table + filesystem blobs |
| Sitemap / seed URLs | `seeds.json` + HuggingFace dataset discovery |
| Crawl DB (Bigtable) | SQLite `crawl.db` (WAL mode) |
| Index builder (offline) | `embed_collection.py` (CLIP) |
| Search serving | `vbaby serve-nft` |

---

## 2. Design Decisions (and Why)

### 2.1 Python for crawling, Rust for search

**Decision:** Crawler is 100% Python. Vector search stays in Rust.

**Why:**
- Crawling is I/O-bound: HTTP, IPFS gateways, JSON parsing, retries. Python's
  ecosystem (urllib, concurrent.futures, sqlite3) is ideal.
- Search is compute-bound: AVX2 distance kernels over millions of vectors.
  Rust gives 10–13× over scalar Python and we already have `FlatIndex`.
- CLIP embedding stays Python (open_clip/torch) — same split as the
  existing BAYC POC.

### 2.2 SQLite for crawl state

**Decision:** Single-file SQLite database at `data/crawl/crawl.db`.

**Why:**
- **Idempotency:** `UNIQUE(chain, contract, token_id)` prevents duplicate work.
  Re-running crawl is safe.
- **Crash recovery:** WAL journal mode — interrupted crawls resume from
  frontier queue.
- **Inspectable:** `sqlite3 data/crawl/crawl.db` for debugging.
- **Scale fit:** Millions of tokens × ~500 bytes/row ≈ hundreds of MB.
  Fine for single-machine crawler. At billions we'd shard by chain or move
  to Postgres/Bigtable.

**Tables:**
| Table | Purpose |
|-------|---------|
| `collections` | Registry of known NFT contracts (like a site list) |
| `tokens` | Per-token state machine + extracted fields |
| `frontier` | Priority queue of pending work |
| `fetch_log` | HTTP audit trail for tuning rate limits |
| `crawl_runs` | Per-session statistics |

### 2.3 Two-stage pipeline: metadata → image

**Decision:** Separate frontier stages for metadata and image fetches.

**Why:**
- Metadata JSON is ~1–5 KB; images are 50 KB–5 MB. Different batch sizes
  and worker counts make sense (`metadata_workers=8`, `image_workers=12`).
- Different hosts: metadata often on IPFS; images may be on Arweave, S3,
  or custom CDNs — separate rate-limit buckets.
- **Re-fetch without re-parse:** If an image 404s but metadata is good,
  we only re-queue the image stage.
- Mirrors Google's separation of HTML document fetch vs embedded resource
  fetch.

### 2.4 Token state machine

```
pending → metadata_fetched → image_fetched → done
                ↓                  ↓
              failed             failed
                ↓
             skipped (no image_uri)
```

**Why:** Every token has exactly one status. Failed tokens retry with
exponential backoff (2^n seconds, cap 300s) up to `max_retries` (default 4).
This prevents hot-looping on permanently broken URIs while giving transient
gateway failures time to recover.

### 2.5 Pluggable metadata providers

**Decision:** `MetadataSource` enum + provider classes per resolution strategy.

| Source | When to use | Example |
|--------|-------------|---------|
| `ipfs_pattern` | Fixed base CID + `/{token_id}` | BAYC |
| `token_uri` | On-chain `tokenURI()` per token | CryptoPunks |
| `huggingface` | Pre-packaged HF parquet datasets | huggingnft/boredapeyachtclub |
| `http_api` | Marketplace indexer APIs | Reservoir, Alchemy |

**Why:** NFT metadata is a mess. BAYC uses a single IPFS folder; other
collections call `tokenURI()` on-chain which returns different URI schemes
per token. A single fetch strategy cannot cover all collections.

### 2.6 IPFS gateway rotation

**Decision:** Try multiple public gateways in order on failure.

```python
GATEWAYS = [
    "https://ipfs.io/ipfs/{path}",
    "https://gateway.pinata.cloud/ipfs/{path}",
    "https://cloudflare-ipfs.com/ipfs/{path}",
    "https://dweb.link/ipfs/{path}",
]
```

**Why:** Public IPFS gateways are unreliable individually but collectively
available. This pattern was proven in `fetch_traits.py` for BAYC. We
generalized it to arbitrary CIDs and paths.

### 2.7 Per-host politeness

**Decision:** `PolitenessManager` enforces `min_delay_per_host` (default 0.25s).

**Why:** Without rate limiting, parallel workers hammer the same gateway
and get 429'd or IP-banned. Google's crawl-delay equivalent. Thread-safe
via lock + monotonic clock.

### 2.8 Crawl / embed separation

**Decision:** Crawler outputs raw images + `manifest.jsonl`. Embedding is a
separate `embed_collection.py` step.

**Why:**
- **Different cadences:** Crawl continuously; re-embed only when CLIP model
  changes.
- **Different resources:** Crawl needs network; embed needs GPU.
- **Idempotent re-embed:** Change model → re-run embed on same manifest
  without re-crawling.
- Same pattern as Google: crawl stores raw HTML; index builder tokenizes
  offline.

### 2.9 Export contract (handoff to search)

Crawler export produces:

```
data/export/<collection>/
  manifest.jsonl     # one JSON object per line, embedder input
  crawl_meta.json    # collection metadata
  images/<tok>.jpg   # flattened thumbnails
```

Embedder produces (unchanged vbaby contract):

```
data/<collection>/
  embeddings.f32     # n×d float32 LE, row i ↔ tokens[i]
  meta.json          # {n, d, model, tokens, collection, contract, chain}
  images/<tok>.jpg
```

**Why:** `FlatIndex::open` only reads `d`, `model`, `tokens` from meta.json.
Extra fields (`collection`, `contract`, `chain`) are forward-compatible via
`#[serde(default)]` — Rust ignores unknown fields on deserialize... actually
Rust serde will error on unknown fields unless we add `#[serde(deny_unknown_fields)]`
is NOT set, so extra fields are fine.

Wait - serde by default ignores unknown fields when deserializing? No - by default serde_json ignores unknown fields only if you don't have deny_unknown_fields. The NftMeta struct only has d, model, tokens - extra fields in JSON are ignored by default in serde. Good.

### 2.10 Seed-driven discovery (not exhaustive chain scan)

**Decision:** Start with curated `seeds.json`, expand via HuggingFace dataset
listing. No full-chain log scanning in v0.1.

**Why:**
- Full Ethereum log scan from block 0 is expensive and noisy.
- High-value collections are known; seeds give immediate utility.
- HF `huggingnft/*` datasets provide pre-validated image+metadata pairs.
- Chain scanning is a future `providers/eth_logs.py` extension when we need
  unknown collection discovery.

---

## 3. Module Reference

```
scripts/
  nft_crawl.py              CLI entry point
  embed_collection.py       Manifest → CLIP → vbaby index
  nft_crawler/
    __init__.py
    models.py               Collection, TokenRecord, enums
    config.py               CrawlerConfig, gateway lists
    db.py                   SQLite schema + CrawlDB
    politeness.py           Per-host rate limiting
    fetcher.py              HTTP fetch + gateway rotation
    resolver.py             ipfs://, ar://, data: URI → HTTP URLs
    parser.py               OpenSea metadata JSON extraction
    discovery.py            Seeds, HF discovery, token enumeration
    crawler.py              Main scheduler (two-stage loop)
    export.py               manifest.jsonl export
    seeds.json              Default seed collections
    providers/
      __init__.py           MetadataProvider implementations
```

---

## 4. Data Layout on Disk

```
data/crawl/
  crawl.db                           # SQLite state
  metadata/ethereum/0xbc4c.../123.json
  images/ethereum/0xbc4c.../123.jpg
  parquet/bayc/                      # optional HF downloads

data/export/bayc/
  manifest.jsonl
  crawl_meta.json
  images/0.jpg, 1.jpg, ...

data/bayc/                           # final search index
  embeddings.f32
  meta.json
  images/0.jpg, ...
```

---

## 5. Usage

```bash
# 1. Initialize DB + register seed collections
python scripts/nft_crawl.py init

# 2. Enqueue tokens (start small)
python scripts/nft_crawl.py seed --collection bayc --limit 50

# 3. Crawl (one batch or until empty)
python scripts/nft_crawl.py crawl --batch-metadata 50 --batch-images 50
python scripts/nft_crawl.py crawl --until-empty

# 4. Check progress
python scripts/nft_crawl.py status --collection bayc

# 5. Export for embedding
python scripts/nft_crawl.py export --collection bayc --out data/export/bayc

# 6. Embed with CLIP
python scripts/embed_collection.py \
  --manifest data/export/bayc/manifest.jsonl \
  --out data/bayc

# 7. Serve semantic search (existing stack)
target/release/vbaby serve-nft --dir data/bayc --port 8091
python scripts/nft_search_app.py --port 8090 --vbaby-port 8091
```

---

## 6. Roadmap to "Exa for NFT Images"

| Phase | What | Status |
|-------|------|--------|
| **0** | BAYC POC (HF parquet → embed → search) | Done |
| **1** | NFT crawler (this PR) | Done |
| **2** | Multi-collection search (merge indices, collection filter) | Next |
| **3** | Reservoir/OpenSea discovery (API-key collection expansion) | Planned |
| **4** | On-chain event indexing (new mints → auto-crawl) | Planned |
| **5** | IVF-PQ index for millions+ vectors | Planned |
| **6** | Continuous crawl + incremental embed pipeline | Planned |

---

## 7. Failure Modes

| Failure | Behavior |
|---------|----------|
| IPFS gateway timeout | Try next gateway; log to `fetch_log` |
| All gateways fail | Re-queue with exponential backoff |
| No `image_uri` in metadata | Status → `skipped` |
| Burned token (empty metadata) | Status → `failed` after retries |
| Rate limit (429) | Politeness delay + backoff |
| Crash mid-crawl | Frontier items already popped are lost for that batch; re-seed re-enqueues only new tokens. Tokens in `metadata_fetched` state resume at image stage on next seed. |

---

## 8. Configuration

Environment variables:
- `ALCHEMY_API_KEY` — enables on-chain `tokenURI` resolution
- `RESERVOIR_API_KEY` — enables Reservoir token metadata API

CLI flags:
- `--data-dir` — crawl root (default `data/crawl`)
- `--batch-metadata` / `--batch-images` — per-cycle throughput
- `--limit` on seed — cap tokens for testing

Custom seeds: JSON file with `collections` array matching `seeds.json` schema.
