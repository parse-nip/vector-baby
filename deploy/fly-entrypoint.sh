#!/usr/bin/env bash
set -euo pipefail
cd /app

DATA_DIR="${DATA_DIR:-/data/bayc}"
PORT_UI="${PORT_UI:-8090}"
PORT_VBABY="${PORT_VBABY:-8091}"
IMAGES="${IMAGES:-$DATA_DIR/images}"
MODEL="${MODEL:-ViT-L-14}"
PRETRAINED="${PRETRAINED:-laion2b_s32b_b82k}"

if [[ ! -f "$DATA_DIR/meta.json" ]]; then
  echo "waiting for BAYC index at $DATA_DIR/meta.json (run fly-upload-data.sh) ..."
  while [[ ! -f "$DATA_DIR/meta.json" ]]; do sleep 10; done
  echo "index found, starting services ..."
fi

vbaby serve-nft --dir "$DATA_DIR" --port "$PORT_VBABY" &
sleep 1

exec python3 scripts/nft_search_app.py \
  --port "$PORT_UI" \
  --vbaby-port "$PORT_VBABY" \
  --data-dir "$DATA_DIR" \
  --images "$IMAGES" \
  --model "$MODEL" \
  --pretrained "$PRETRAINED" \
  --device cpu
