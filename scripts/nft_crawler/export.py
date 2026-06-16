"""Export crawled data to vector-baby flat-index ingest format.

Bridge between crawler output and embed_collection.py:
  data/crawl/images/ethereum/0xabc.../123.jpg
    → data/export/bayc/images/123.jpg + manifest.jsonl

The manifest is the handoff contract for the embedding pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import CrawlDB


def export_collection(
    db: CrawlDB,
    collection_slug: str,
    out_dir: str,
    status_filter: str = "done",
) -> dict:
    """Export done tokens to flat-index-ready directory layout."""
    col = db.get_collection(collection_slug)
    if not col:
        raise ValueError(f"unknown collection: {collection_slug}")

    conn = db.connect()
    rows = conn.execute(
        """
        SELECT token_id, name, image_path, image_sha256, traits, image_uri
        FROM tokens
        WHERE collection_slug=? AND status=?
        ORDER BY token_id
        """,
        (collection_slug, status_filter),
    ).fetchall()

    img_out = os.path.join(out_dir, "images")
    os.makedirs(img_out, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    tokens = []
    exported = 0
    with open(manifest_path, "w") as mf:
        for row in rows:
            if not row["image_path"] or not os.path.exists(row["image_path"]):
                continue
            tok = row["token_id"]
            ext = os.path.splitext(row["image_path"])[1] or ".jpg"
            dest = os.path.join(img_out, f"{tok}{ext}")
            if not os.path.exists(dest):
                shutil.copy2(row["image_path"], dest)
            entry = {
                "token_id": tok,
                "name": row["name"],
                "image_path": dest,
                "image_sha256": row["image_sha256"],
                "traits": json.loads(row["traits"] or "{}"),
                "image_uri": row["image_uri"],
                "collection": collection_slug,
                "contract": col.contract,
                "chain": col.chain.value,
            }
            mf.write(json.dumps(entry) + "\n")
            tokens.append(tok)
            exported += 1

    # Write collection meta for embedder (not the final meta.json — embedder produces that)
    crawl_meta = {
        "collection": collection_slug,
        "name": col.name,
        "chain": col.chain.value,
        "contract": col.contract,
        "n_exported": exported,
        "tokens": tokens,
        "model_hint": "ViT-L-14/laion2b_s32b_b82k",
    }
    with open(os.path.join(out_dir, "crawl_meta.json"), "w") as f:
        json.dump(crawl_meta, f, indent=2)

    return {"exported": exported, "out_dir": out_dir, "manifest": manifest_path}
