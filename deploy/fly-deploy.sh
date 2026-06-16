#!/usr/bin/env bash
# First-time Fly.io deploy for noogle (BAYC semantic search).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.fly/bin:${PATH}"

APP="${FLY_APP:-noogle}"
REGION="${FLY_REGION:-sjc}"

if ! fly auth whoami &>/dev/null; then
  echo "not logged in — run: fly auth login"
  exit 1
fi

if [[ ! -f data/bayc/meta.json ]]; then
  echo "missing data/bayc — run: python3 scripts/embed_bayc.py --download"
  exit 1
fi

if ! fly apps list 2>/dev/null | grep -q "^${APP}\b"; then
  echo "creating app $APP in $REGION ..."
  fly apps create "$APP" || true
fi

if ! fly volumes list -a "$APP" 2>/dev/null | grep -q noogle_data; then
  echo "creating volume noogle_data (2GB) ..."
  fly volumes create noogle_data --region "$REGION" --size 2 -a "$APP" -y
fi

echo "deploying ..."
fly deploy -a "$APP" --config deploy/fly.toml

echo "uploading BAYC index to volume ..."
"$ROOT/deploy/fly-upload-data.sh" "$APP"

echo ""
echo "=== next: custom domain (Fly TLS + Cloudflare DNS) ==="
echo "  fly certs add noogle.popped.dev -a $APP"
echo ""
echo "Then in Cloudflare DNS for popped.dev:"
echo "  Type: CNAME   Name: noogle   Target: ${APP}.fly.dev   Proxy: DNS only (grey cloud)"
echo ""
echo "Open: https://noogle.popped.dev  (after cert validates, ~1–5 min)"
