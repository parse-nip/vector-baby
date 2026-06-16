#!/usr/bin/env python3
"""Bored Ape Search web app.

Pipeline per query:
  text  --(CLIP text encoder)-->  vector  --(POST to vector-baby)-->  token ids
        --> render ape thumbnails.

The vector search itself runs in the Rust `vbaby serve-nft` service; this app
only does CLIP text encoding, the web UI, and image serving.
"""
import argparse, json, os, time, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import torch, open_clip

ARGS = None
MODEL = None
TOKENIZER = None
DEVICE = None
BASELINE = None
BASELINE_ALPHA = 0.5
MODEL_READY = threading.Event()

PAGE = """<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Bored Ape Search — vector-baby</title>
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
.meta{color:#8a94a6;font-size:13px;margin-bottom:8px;min-height:18px}
.meta b{color:#7ee787}
.timing{margin-bottom:14px}
.timing-labels{display:flex;justify-content:space-between;font-size:11px;color:#8a94a6;margin-bottom:5px;gap:12px;flex-wrap:wrap}
.timing-labels span{display:flex;align-items:center;gap:5px}
.dot{width:8px;height:8px;border-radius:2px;display:inline-block}
.dot.encode{background:#f59e0b}.dot.search{background:#38bdf8}
.timing-track{display:flex;height:10px;border-radius:6px;overflow:hidden;background:#161b26;border:1px solid #1f2733}
.timing-seg{height:100%;transition:width .35s ease}
.timing-seg.encode{background:linear-gradient(90deg,#d97706,#fbbf24)}
.timing-seg.search{background:linear-gradient(90deg,#0284c7,#38bdf8)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}
.card{background:#121722;border:1px solid #1f2733;border-radius:12px;overflow:hidden}
.card img{width:100%;display:block;aspect-ratio:1;object-fit:cover;background:#0b0e14}
.card .info{padding:8px 10px;font-size:12px;display:flex;justify-content:space-between}
.card .score{color:#7ee787}
</style></head><body>
<h1>Bored Ape Search</h1>
<div class=sub>type a concept &middot; CLIP text&rarr;image &middot; exact neural search over 10,000 BAYC apes via vector-baby</div>
<div class=bar>
  <input id=q placeholder="golden fur ape" autofocus>
  <button onclick=run()>Search</button>
</div>
<div class=chips id=chips></div>
<div class=meta id=meta></div>
<div class=timing id=timing style=display:none>
  <div class=timing-labels>
    <span><i class=dot style=background:#fbbf24></i> text encode <b id=lab-enc>—</b></span>
    <span><i class=dot style=background:#38bdf8></i> vector search <b id=lab-srch>—</b></span>
    <span>total <b id=lab-tot>—</b></span>
  </div>
  <div class=timing-track>
    <div class="timing-seg encode" id=seg-enc></div>
    <div class="timing-seg search" id=seg-srch></div>
  </div>
</div>
<div class=grid id=grid></div>
<script>
const examples=["golden fur","laser eyes","party hat","sunglasses","zombie","robot","wizard hat","diamond grill","crown","blue fur","smoking a cigar","cyborg"];
const chips=document.getElementById('chips');
examples.forEach(e=>{const c=document.createElement('span');c.className='chip';c.textContent=e;c.onclick=()=>{document.getElementById('q').value=e;run()};chips.appendChild(c)});

function setTiming(d){
  const enc=d.encode_ms, srch=d.search_ms, tot=d.total_ms;
  const encPct=Math.max(0.4,enc/tot*100), srchPct=Math.max(0.4,srch/tot*100);
  document.getElementById('meta').innerHTML=`"${d.query}" &mdash; <b>${tot.toFixed(1)} ms</b> total (text encode ${enc.toFixed(1)} ms + vector search ${srch.toFixed(2)} ms over ${d.n.toLocaleString()} apes)`;
  document.getElementById('timing').style.display='block';
  document.getElementById('lab-enc').textContent=enc.toFixed(1)+' ms ('+encPct.toFixed(0)+'%)';
  document.getElementById('lab-srch').textContent=srch.toFixed(2)+' ms ('+srchPct.toFixed(0)+'%)';
  document.getElementById('lab-tot').textContent=tot.toFixed(1)+' ms';
  document.getElementById('seg-enc').style.width=encPct+'%';
  document.getElementById('seg-srch').style.width=srchPct+'%';
}

async function run(){
  const q=document.getElementById('q').value.trim();if(!q)return;
  document.getElementById('meta').textContent='searching...';
  document.getElementById('timing').style.display='none';
  const r=await fetch('/search?k=24&q='+encodeURIComponent(q));const d=await r.json();
  setTiming(d);
  document.getElementById('grid').innerHTML=d.results.map(x=>
    `<div class=card><img loading=lazy src="/img/${x.token}.jpg"><div class=info><span>#${x.token}</span><span class=score>${x.score.toFixed(3)}</span></div></div>`
  ).join('');
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')run()});
</script></body></html>"""


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _embed(text):
    with torch.no_grad():
        toks = TOKENIZER([text]).to(DEVICE)
        f = MODEL.encode_text(toks)
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
        if u.path == "/health":
            code = 200 if MODEL_READY.is_set() else 503
            body = "ok" if code == 200 else "loading"
            return self._send(code, body, "text/plain")
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
            if not MODEL_READY.is_set():
                return self._send(503, json.dumps({"error": "loading model, try again"}))
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


def boot_clip():
    global MODEL, TOKENIZER, DEVICE, NUM_VECTORS, BASELINE
    try:
        info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{ARGS.vbaby_port}/api/info", timeout=30).read())
        NUM_VECTORS = info.get("n", 0)
    except Exception as e:
        print("warning: could not reach vbaby:", e, flush=True)
    DEVICE = pick_device(ARGS.device)
    print(f"loading CLIP ({ARGS.model}) on {DEVICE} ...", flush=True)
    MODEL, _, _ = open_clip.create_model_and_transforms(ARGS.model, pretrained=ARGS.pretrained)
    MODEL.eval().to(DEVICE)
    TOKENIZER = open_clip.get_tokenizer(ARGS.model)
    BASELINE = _embed(ARGS.baseline_text) if ARGS.baseline_text else None
    MODEL_READY.set()
    print(f"ready — {NUM_VECTORS} vectors on {DEVICE}", flush=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--vbaby-port", type=int, default=8091)
    ap.add_argument("--data-dir", default="data/bayc", help="BAYC index dir (images/)")
    ap.add_argument("--images", default=None, help="thumbnail dir (default: <data-dir>/images)")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--baseline-text", default="a bored ape", help="prompt subtracted to isolate fine attributes; empty to disable")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ARGS = ap.parse_args()
    if ARGS.images is None:
        ARGS.images = os.path.join(ARGS.data_dir, "images")
    threading.Thread(target=boot_clip, daemon=True).start()
    print(f"Bored Ape Search on http://0.0.0.0:{ARGS.port}  (vbaby on {ARGS.vbaby_port}, CLIP loading in background)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
