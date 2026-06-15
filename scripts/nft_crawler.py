#!/usr/bin/env python3
"""A small but real NFT crawler: OpenSea discovery -> CDN image fetch -> dedup
-> CLIP embed -> incremental index for vector-baby.

Architecture (see AGENTS.md):
  producer thread  : paginate OpenSea collections, rate-limited (60/min, 429-aware)
  fetcher threads  : download CDN image, dedup by (contract:token) and image sha256,
                     CLIP-preprocess
  main loop        : batch CLIP encode_image, append embeddings.f32 + docs.jsonl,
                     checkpoint meta.json + crawl state (resumable)
"""
import argparse, hashlib, io, json, os, queue, threading, time, urllib.error, urllib.request
import numpy as np
import torch, open_clip
from PIL import Image

API = "https://api.opensea.io/api/v2"
OUT = "data/crawl"

# Seed frontier: a diverse set of well-known collections so cross-collection
# semantic search is interesting. Unknown slugs are skipped gracefully.
SEED = [
    "boredapeyachtclub", "azuki", "pudgypenguins", "doodles-official",
    "proof-moonbirds", "cool-cats-nft", "mfers", "world-of-women-nft",
    "cyberkongz", "cryptoadz-by-gremplin", "lazy-lions", "meebits",
    "deadfellaz", "cryptodickbutts-s3", "the-doge-pound", "0n1-force",
]

_rate_lock = threading.Lock()
_last_call = [0.0]
COUNTS = {"discovered": 0, "fetched": 0, "embedded": 0, "dup": 0, "err": 0}
_clock = threading.Lock()


def load_key():
    if os.environ.get("OPENSEA_API_KEY"):
        return os.environ["OPENSEA_API_KEY"]
    return json.load(open(f"{OUT}/api_key.json"))["api_key"]


def rate_gate(min_interval):
    # Token-bucket-ish: ensure a minimum gap between OpenSea API calls so we
    # stay under 60/min and never trip 429 in the steady state.
    with _rate_lock:
        dt = time.time() - _last_call[0]
        if dt < min_interval:
            time.sleep(min_interval - dt)
        _last_call[0] = time.time()


UA = "Mozilla/5.0 (X11; Linux x86_64) vector-baby-crawler/0.1"


def http_get(url, headers=None, retries=5):
    headers = dict(headers or {})
    headers.setdefault("User-Agent", UA)  # OpenSea WAF 403s the default python UA
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "3")) + 0.5
                time.sleep(wait)
                continue
            if e.code in (404, 400):
                return e.code, b"", {}
            time.sleep(1.0 * (i + 1))
        except Exception:
            time.sleep(1.0 * (i + 1))
    return 0, b"", {}


def producer(key, recordq, args, state):
    headers = {"X-API-KEY": key, "accept": "application/json"}
    for slug in SEED:
        cstate = state["collections"].get(slug, {})
        cur = cstate.get("cursor")
        fetched = cstate.get("fetched", 0)
        if cstate.get("done"):
            continue
        while fetched < args.per_collection:
            rate_gate(args.min_interval)
            url = f"{API}/collection/{slug}/nfts?limit={args.page}"
            if cur:
                url += f"&next={cur}"
            st, body, _ = http_get(url, headers)
            if st != 200:
                print(f"[producer] {slug}: stop (http {st})", flush=True)
                break
            d = json.loads(body)
            nfts = d.get("nfts", [])
            for n in nfts:
                recordq.put((slug, n))
            with _clock:
                COUNTS["discovered"] += len(nfts)
            fetched += len(nfts)
            cur = d.get("next")
            state["collections"][slug] = {"cursor": cur, "fetched": fetched, "done": not cur}
            save_state(state)
            if not cur:
                break
        print(f"[producer] {slug}: discovered {fetched}", flush=True)
    for _ in range(args.workers):
        recordq.put(None)


def fetcher(recordq, embedq, preprocess, seen, seen_lock):
    while True:
        item = recordq.get()
        if item is None:
            embedq.put(None)
            return
        slug, n = item
        token = n.get("identifier")
        contract = n.get("contract")
        img_url = n.get("display_image_url") or n.get("image_url")
        if not img_url or token is None:
            continue
        key = f"{contract}:{token}"
        with seen_lock:
            if key in seen:
                with _clock: COUNTS["dup"] += 1
                continue
            seen.add(key)
        st, body, _ = http_get(img_url, retries=3)
        if st != 200 or not body:
            with _clock: COUNTS["err"] += 1
            continue
        h = hashlib.sha256(body).hexdigest()
        with seen_lock:
            if h in seen:  # same art, different token (copymint) -> drop
                with _clock: COUNTS["dup"] += 1
                continue
            seen.add(h)
        try:
            img = Image.open(io.BytesIO(body)).convert("RGB")
            t = preprocess(img)
        except Exception:
            with _clock: COUNTS["err"] += 1
            continue
        meta = {
            "chain": "ethereum", "contract": contract, "token": token,
            "name": n.get("name") or f"{slug} #{token}",
            "collection": slug, "image_url": img_url,
        }
        with _clock: COUNTS["fetched"] += 1
        embedq.put((meta, t))


def save_state(state):
    tmp = f"{OUT}/state.json.tmp"
    json.dump(state, open(tmp, "w"))
    os.replace(tmp, f"{OUT}/state.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-collection", type=int, default=300)
    ap.add_argument("--page", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-interval", type=float, default=1.1)  # ~55 req/min < 60
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    os.makedirs(f"{OUT}/images", exist_ok=True)
    emb_path, docs_path, meta_path = f"{OUT}/embeddings.f32", f"{OUT}/docs.jsonl", f"{OUT}/meta.json"
    state_path = f"{OUT}/state.json"

    if args.reset:
        for p in [emb_path, docs_path, meta_path, state_path, f"{OUT}/seen.json"]:
            if os.path.exists(p): os.remove(p)

    state = json.load(open(state_path)) if os.path.exists(state_path) else {"collections": {}}
    seen = set(json.load(open(f"{OUT}/seen.json"))) if os.path.exists(f"{OUT}/seen.json") else set()
    n_total = sum(1 for _ in open(docs_path)) if os.path.exists(docs_path) else 0
    print(f"resuming with {n_total} already indexed, {len(seen)} seen keys", flush=True)

    torch.set_num_threads(os.cpu_count() or 4)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.eval()
    d = model.visual.output_dim
    print(f"crawler ready: model={args.model} d={d}", flush=True)

    key = load_key()
    recordq = queue.Queue(maxsize=4000)
    embedq = queue.Queue(maxsize=2000)
    seen_lock = threading.Lock()

    prod = threading.Thread(target=producer, args=(key, recordq, args, state), daemon=True)
    prod.start()
    workers = []
    for _ in range(args.workers):
        w = threading.Thread(target=fetcher, args=(recordq, embedq, preprocess, seen, seen_lock), daemon=True)
        w.start()
        workers.append(w)

    emb_f = open(emb_path, "ab")
    docs_f = open(docs_path, "a")
    buf_meta, buf_t = [], []
    done = 0
    t0 = time.time()

    def flush():
        nonlocal n_total
        if not buf_t:
            return
        with torch.no_grad():
            x = torch.stack(buf_t)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        arr = f.cpu().numpy().astype(np.float32)
        emb_f.write(arr.tobytes()); emb_f.flush()
        for m in buf_meta:
            docs_f.write(json.dumps(m) + "\n")
        docs_f.flush()
        n_total += len(buf_meta)
        with _clock: COUNTS["embedded"] = n_total
        json.dump({"d": d, "n": n_total, "model": args.model, "tokens": list(range(n_total))}, open(meta_path, "w"))
        json.dump(list(seen), open(f"{OUT}/seen.json", "w"))
        buf_meta.clear(); buf_t.clear()
        rate = n_total / (time.time() - t0 + 1e-9)
        print(f"[index] n={n_total} | discovered={COUNTS['discovered']} fetched={COUNTS['fetched']} "
              f"dup={COUNTS['dup']} err={COUNTS['err']} | {rate:.1f}/s", flush=True)

    while done < args.workers:
        item = embedq.get()
        if item is None:
            done += 1
            continue
        meta, t = item
        buf_meta.append(meta); buf_t.append(t)
        if len(buf_t) >= args.batch:
            flush()
    flush()
    emb_f.close(); docs_f.close()
    print(f"CRAWL COMPLETE: indexed {n_total} NFTs across {len([c for c in state['collections'] if state['collections'][c].get('fetched')])} collections in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
