# syntax=docker/dockerfile:1

# --- Rust vector search binary ---
FROM rust:1-bookworm AS rust
WORKDIR /app
COPY .cargo .cargo
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# --- Runtime: Python (CLIP) + vbaby ---
FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/requirements.txt scripts/
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r scripts/requirements.txt

COPY --from=rust /app/target/release/vbaby /usr/local/bin/vbaby
COPY scripts/nft_search_app.py scripts/
COPY deploy/fly-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Index lives on a Fly volume at /data (see fly.toml)
ENV DATA_DIR=/data/bayc
ENV PORT_UI=8090
ENV PORT_VBABY=8091

EXPOSE 8090
ENTRYPOINT ["/entrypoint.sh"]
