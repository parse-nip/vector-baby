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
