#!/usr/bin/env python3
"""Semantic NFT search web app (BAYC POC).

Pipeline per query:
  text  --(CLIP text encoder)-->  vector  --(POST to vector-baby)-->  token ids
        --> render ape thumbnails.

The vector search itself runs in the Rust `vbaby serve-nft` service; this app
only does CLIP text encoding, the web UI, and image serving.
"""
import argparse, json, os, time, urllib.request, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import torch, open_clip

ARGS = None
MODEL = None
TOKENIZER = None
BASELINE = None  # canonical "a bored ape" embedding, subtracted to isolate the
                 # distinctive part of a query (helps fine attributes like fur color)
BASELINE_ALPHA = 0.5

PAGE = """<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>NFT semantic search — vector-baby</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;font-family:ui-monospace,Menlo,monospace;background:#0b0e14;color:#e6e6e6;padding:28px}
h1{font-size:22px;margin:0 0 2px}.sub{color:#8a94a6;font-size:13px;margin-bottom:18px}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
input{background:#121722;border:1px solid #1f2733;color:#e6e6e6;border-radius:10px;padding:12px 14px;font:inherit;width:380px}
button{background:#2563eb;color:#fff;border:0;border-radius:10px;padding:12px 18px;font:inherit;cursor:pointer}
button:hover{background:#1d4ed8}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.chip{background:#161b26;border:1px solid #1f2733;color:#9fb4d6;border-radius:999px;padding:6px 12px;font-size:12px;cursor:pointer}
.chip:hover{background:#1f2733}
.meta{color:#8a94a6;font-size:13px;margin-bottom:14px;min-height:18px}
.meta b{color:#7ee787}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}
.card{background:#121722;border:1px solid #1f2733;border-radius:12px;overflow:hidden}
.card img{width:100%;display:block;aspect-ratio:1;object-fit:cover;background:#0b0e14}
.card .info{padding:8px 10px;font-size:12px;display:flex;justify-content:space-between}
.card .score{color:#7ee787}
</style></head><body>
<h1>NFT semantic search</h1>
<div class=sub>type a concept &middot; CLIP text&rarr;image &middot; exact cosine search over 10,000 Bored Apes via vector-baby</div>
<div class=bar>
  <input id=q placeholder="golden fur ape" autofocus>
  <button onclick=run()>Search</button>
</div>
<div class=chips id=chips></div>
<div class=meta id=meta></div>
<div class=grid id=grid></div>
<script>
const examples=["golden fur ape","zombie ape","laser eyes","ape wearing a captain's hat","ape smoking a cigarette","ape with a king's crown","robot ape","ape wearing 3D glasses","rainbow colored ape","ape in a suit and tie"];
const chips=document.getElementById('chips');
examples.forEach(e=>{const c=document.createElement('span');c.className='chip';c.textContent=e;c.onclick=()=>{document.getElementById('q').value=e;run()};chips.appendChild(c)});
async function run(){
  const q=document.getElementById('q').value.trim();if(!q)return;
  document.getElementById('meta').textContent='searching...';
  const r=await fetch('/search?k=24&q='+encodeURIComponent(q));const d=await r.json();
  document.getElementById('meta').innerHTML=`"${d.query}" &mdash; <b>${d.total_ms.toFixed(1)} ms</b> total (text encode ${d.encode_ms.toFixed(1)} ms + vector search ${d.search_ms.toFixed(2)} ms over ${d.n.toLocaleString()} apes)`;
  document.getElementById('grid').innerHTML=d.results.map(x=>
    `<div class=card><img loading=lazy src="/img/${x.token}.jpg"><div class=info><span>#${x.token}</span><span class=score>${x.score.toFixed(3)}</span></div></div>`).join('');
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')run()});
</script></body></html>"""


def _embed(text):
    with torch.no_grad():
        f = MODEL.encode_text(TOKENIZER([text]))
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().numpy().astype("float32")


def encode_text(q):
    t = time.time()
    f = _embed(q)
    if BASELINE is not None:
        f = f - BASELINE_ALPHA * BASELINE
        f = f / (float((f * f).sum()) ** 0.5 + 1e-9)
    return f.tolist(), (time.time() - t) * 1000.0


def vbaby_search(vec, k):
    req = urllib.request.Request(
        f"http://127.0.0.1:{ARGS.vbaby_port}/api/search_vector",
        data=json.dumps({"vector": vec, "k": k}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path.startswith("/img/"):
            name = os.path.basename(u.path)
            fp = os.path.join(ARGS.images, name)
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    return self._send(200, fh.read(), "image/jpeg")
            return self._send(404, b"no image", "text/plain")
        if u.path == "/search":
            qs = parse_qs(u.query)
            q = (qs.get("q", [""])[0]).strip()
            k = int(qs.get("k", ["24"])[0])
            if not q:
                return self._send(400, json.dumps({"error": "empty query"}))
            t0 = time.time()
            vec, enc_ms = encode_text(q)
            sr = vbaby_search(vec, k)
            total = (time.time() - t0) * 1000.0
            out = {
                "query": q,
                "encode_ms": enc_ms,
                "search_ms": sr.get("latency_ms", 0.0),
                "total_ms": total,
                "n": NUM_VECTORS,
                "results": sr.get("results", []),
            }
            return self._send(200, json.dumps(out))
        return self._send(404, b"not found", "text/plain")


NUM_VECTORS = 0


def main():
    global ARGS, MODEL, TOKENIZER, NUM_VECTORS, BASELINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--vbaby-port", type=int, default=8091)
    ap.add_argument("--images", default="data/bayc/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ARGS = ap.parse_args()
    torch.set_num_threads(2)
    MODEL, _, _ = open_clip.create_model_and_transforms(ARGS.model, pretrained=ARGS.pretrained)
    MODEL.eval()
    TOKENIZER = open_clip.get_tokenizer(ARGS.model)
    BASELINE = _embed("a bored ape")
    try:
        info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{ARGS.vbaby_port}/api/info", timeout=10).read())
        NUM_VECTORS = info.get("n", 0)
    except Exception as e:
        print("warning: could not reach vbaby:", e)
    print(f"NFT search UI on http://0.0.0.0:{ARGS.port}  (vbaby on {ARGS.vbaby_port}, {NUM_VECTORS} vectors)")
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
