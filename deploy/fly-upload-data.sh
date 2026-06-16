#!/usr/bin/env bash
# Upload BAYC index artifacts (embeddings + thumbnails) to a Fly volume.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-noogle}"
SRC="$ROOT/data/bayc"

export PATH="${HOME}/.fly/bin:${PATH}"

for f in embeddings.f32 meta.json; do
  [[ -f "$SRC/$f" ]] || { echo "missing $SRC/$f — run embed_bayc.py first"; exit 1; }
done
[[ -d "$SRC/images" ]] || { echo "missing $SRC/images"; exit 1; }

fly machine start -a "$APP" 2>/dev/null || true
sleep 3

TMP="$(mktemp /tmp/bayc-index.XXXXXX.tar.gz)"
trap 'rm -f "$TMP"' EXIT
echo "packing index ..."
tar czf "$TMP" -C "$SRC" embeddings.f32 meta.json images

echo "uploading to fly app=$APP ..."
fly ssh console -a "$APP" -C "sh -c 'mkdir -p /data/bayc'"
fly ssh sftp put "$TMP" /tmp/bayc-index.tar.gz -a "$APP"
fly ssh console -a "$APP" -C "sh -c 'tar xzf /tmp/bayc-index.tar.gz -C /data/bayc && rm /tmp/bayc-index.tar.gz'"

echo "done. verify:"
fly ssh console -a "$APP" -C "sh -c 'ls -la /data/bayc && wc -c /data/bayc/embeddings.f32'"
