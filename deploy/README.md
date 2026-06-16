# Deploy noogle.popped.dev — Fly.io + Cloudflare DNS

Fly runs the app; Cloudflare manages DNS for `popped.dev`. This is the standard combo.

## Architecture

```
User → noogle.popped.dev
     → Cloudflare DNS (CNAME, grey cloud)
     → Fly.io edge (TLS terminated here)
     → Machine: Python UI :8090 + Rust vbaby :8091
     → Volume /data/bayc (embeddings + thumbnails)
```

Cloudflare does **not** proxy traffic (grey cloud). Fly handles HTTPS. You still get Cloudflare DNS, analytics, and other CF services on the zone.

## 0. CLI (installed once)

```bash
# already installed to ~/.fly/bin — add to ~/.zshrc:
export PATH="$HOME/.fly/bin:$PATH"

fly auth login
```

## 1. Build the BAYC index locally (one-time)

```bash
pip install torch torchvision
pip install -r scripts/requirements.txt
python3 scripts/embed_bayc.py --download --device mps   # or cuda
```

## 2. Deploy to Fly

```bash
./deploy/fly-deploy.sh
```

This creates the app, a 2 GB volume, deploys the Docker image, and uploads `data/bayc/` (embeddings + images, not parquet).

## 3. Custom domain — Fly + Cloudflare DNS

```bash
fly certs add noogle.popped.dev -a noogle
fly certs show noogle.popped.dev -a noogle   # wait until Status = Ready
```

In **Cloudflare Dashboard** → `popped.dev` → **DNS**:

| Type  | Name   | Content         | Proxy status      |
|-------|--------|-----------------|-------------------|
| CNAME | noogle | `noogle.fly.dev`| **DNS only** (grey cloud) |

**Important:** use grey cloud, not orange. Orange cloud (proxied) double-terminates TLS and breaks Fly's automatic certs.

SSL/TLS mode in Cloudflare can stay at default — it doesn't apply when DNS-only.

## 4. Verify

```bash
fly status -a noogle
fly logs -a noogle
curl -s https://noogle.popped.dev/search?k=3&q=golden+fur | head
```

## Costs (approx)

- **4 GB RAM** shared-cpu machine: ~$10–15/mo always-on
- **2 GB volume**: ~$0.30/mo
- Cloudflare DNS: free

## Re-upload index after re-embedding

```bash
./deploy/fly-upload-data.sh noogle
fly machines restart -a noogle
```

## Orange cloud (optional, advanced)

If you want Cloudflare proxy (DDoS, WAF) in front of Fly, you need extra setup (origin certs, SSL Full). Grey cloud is strongly recommended for first deploy.
