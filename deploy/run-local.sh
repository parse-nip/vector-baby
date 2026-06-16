#!/usr/bin/env bash
# Start BAYC semantic search (Rust vector search + Python UI/CLIP).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/bayc}"
PORT_UI="${PORT_UI:-8090}"
PORT_VBABY="${PORT_VBABY:-8091}"
IMAGES="${IMAGES:-$DATA_DIR/images}"
MODEL="${MODEL:-ViT-L-14}"
PRETRAINED="${PRETRAINED:-laion2b_s32b_b82k}"

if [[ ! -f "$DATA_DIR/meta.json" ]]; then
  echo "missing index at $DATA_DIR — run: python3 scripts/embed_bayc.py --download"
  exit 1
fi

if [[ ! -f target/release/vbaby ]]; then
  echo "building vbaby (release)..."
  cargo build --release
fi

cleanup() {
  kill "$VBABY_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

./target/release/vbaby serve-nft --dir "$DATA_DIR" --port "$PORT_VBABY" &
VBABY_PID=$!
sleep 0.5

python3 scripts/nft_search_app.py \
  --port "$PORT_UI" \
  --vbaby-port "$PORT_VBABY" \
  --images "$IMAGES" \
  --model "$MODEL" \
  --pretrained "$PRETRAINED" &
UI_PID=$!

echo "UI:    http://127.0.0.1:$PORT_UI"
echo "vbaby: http://127.0.0.1:$PORT_VBABY"
wait
