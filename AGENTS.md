# vector-baby

A custom, from-scratch **billion-scale vector database** in Rust: IVF-PQ
(inverted file + product quantization), disk-backed via mmap, with exact
re-ranking. Single binary `vbaby` with subcommands `build`, `bench`, `recall`,
`serve`, `diag`.

## Cursor Cloud specific instructions

### Build / run
- Always build release: `cargo build --release`. The debug build is orders of
  magnitude too slow to run any real workload — never benchmark a debug build.
- Binary is `target/release/vbaby`. Subcommands:
  - `build  --dir <d> --n <count>` — build an index (synthetic data, generated
    deterministically from the vector id; raw vectors are never stored).
  - `bench  --dir <d> [--nq --nprobe --k --rerank --warmup]` — latency + planted recall.
  - `recall --dir <d> [--nq --nprobe --k --rerank]` — recall vs **exact brute
    force**; only feasible up to ~10M (it scans everything).
  - `serve  --dir <d> [--port 8080]` — HTTP server + single-page UI for live demos.
  - `diag   --dir <d> [--n --batch]` — per-stage build throughput attribution.

### Non-obvious gotchas (read before touching the kernels or config)
- **SIMD feature flags are load-bearing.** `.cargo/config.toml` sets
  `target-feature=+avx2,+fma`. The hot distance kernels in `src/math.rs` are
  gated on `#[cfg(target_feature = "avx2")]`; if that cfg is false they silently
  compile out and a ~10–13x slower scalar fallback runs. `target-cpu=native`
  alone does NOT reliably enable these on this hypervisor, so the explicit flags
  must stay. (AVX-512 is present on the host but its intrinsics are unstable on
  the pinned stable Rust, so we deliberately use AVX2.)
- **Recall comes from re-ranking, not from PQ alone.** IVF-PQ is only a fast
  filter; with 8-byte codes its raw top-k recall is mediocre and *degrades as
  list length grows*. The `--rerank N` step recomputes exact L2 on the top-N
  candidate shortlist (regenerating each candidate vector by id, the stand-in
  for "read original vector from disk") and yields recall@10 = 100%. Increasing
  `nprobe` alone does little — the true neighbors cluster into a few lists, so
  shortlist size + re-rank is the lever.

### Scale / resources
- Index files live under `data/` (gitignored). A 1B index is ~8 GB codes +
  ~4 GB ids on disk, plus a ~4 GB `listid.tmp` during build (deleted after).
- The 1B build uses two streaming passes (assign, then encode). Approximate
  times on this 4-core box: 1M ~15s, 10M ~35s, 1B ~40–50 min.
- `bench`/`serve` call `warmup()` to page the mmap'd code store into the page
  cache; the 8 GB code store fits in the 15 GB RAM, so warm queries are
  RAM-speed. Cold (first) queries hit disk and are slower.

### NFT semantic search POC (BAYC)
- Goal: text query ("golden fur ape") → matching Bored Ape images, <100 ms.
- Two-service architecture (kept separate because CLIP is Python):
  1. **Rust** `vbaby serve-nft --dir data/bayc --port 8091` — loads the
     `FlatIndex` (`src/flat.rs`): exact cosine search over the CLIP image
     embeddings. Exposes `POST /api/search_vector` ({"vector":[...],"k":N}).
     Flat/exact is the right choice at 10k (perfect recall, ~1 ms); IVF-PQ is
     only needed at the billion scale.
  2. **Python** `scripts/nft_search_app.py --port 8090 --vbaby-port 8091` —
     loads CLIP, serves the web UI + ape thumbnails, encodes the text query,
     and forwards the vector to the Rust service.
- CLIP embeddings are L2-normalized, so cosine ranking == L2 ranking; the
  existing `l2_sq` kernel is reused unchanged (cosine reported as 1 - l2/2).
- Offline embedding: `scripts/embed_bayc.py` reads HF parquet (BAYC mirror
  `huggingnft/boredapeyachtclub`), writes `data/bayc/{embeddings.f32,meta.json}`
  and `images/<token>.jpg`. **Model choice matters for fine attributes**:
  ViT-B-32 misses rare traits like solid-gold fur; ViT-L-14 is much better
  (set `--model ViT-L-14 --pretrained laion2b_s32b_b82k`). The text encoder in
  the serving app MUST use the same model as the image embeddings.
- Python deps are a POC add-on (not in the Rust build / update script); install
  per `scripts/requirements.txt` (use `pip install --break-system-packages`,
  no venv available on this VM).
- The app subtracts a canonical baseline embedding (`emb("a bored ape")*0.5`)
  from each query before searching; this cancels the generic "ape" signal and
  sharply improves rare fine-attribute queries (e.g. "golden fur" stops
  returning gold-*chain* apes). Verified not to hurt clean queries.
- **Latency breakdown** (CPU-only box): the vector search itself is ~2 ms; the
  CLIP **text encoder dominates** (~60–90 ms for ViT-L-14, ~10 ms for ViT-B-32)
  and is the only thing near the 100 ms budget — it would be a few ms on a GPU.
  `scripts/eval_query.py` renders a montage of top results for fast visual
  iteration; `scripts/find_gold.py` is local pixel-based ground truth.

### NFT crawler (OpenSea → CLIP → index)
- `scripts/nft_crawler.py` is a real crawler: producer thread paginates OpenSea
  collections (rate-limited, 429-aware), fetcher threads download CDN images +
  dedup (by `contract:token` and image sha256), main loop batches CLIP
  `encode_image` and appends `data/crawl/{embeddings.f32,docs.jsonl,meta.json}`.
  Resumable via `data/crawl/state.json` + `seen.json` (use `--reset` to start
  fresh). Discovery is built on OpenSea's index (CDN images) rather than raw
  chain+IPFS — IPFS gateways rate-limit hard, OpenSea already normalized it.
- **Gotcha**: OpenSea's WAF returns 403 to the default Python user-agent — the
  crawler sets a browser-like `User-Agent` on every request. Get a free instant
  key via `curl -X POST https://api.opensea.io/api/v2/auth/keys` (60 reads/min,
  30-day expiry); cached to `data/crawl/api_key.json` (gitignored) or set
  `OPENSEA_API_KEY`. Don't recreate keys (3/hour/IP limit).
- Serve the crawl: `vbaby serve-nft --dir data/crawl --port 8091` +
  `nft_search_app.py --docs data/crawl/docs.jsonl --model ViT-B-32 --baseline-text ""`
  (disable the BAYC-specific baseline for multi-collection corpora; the UI then
  renders remote CDN images and collection labels from the docstore).
