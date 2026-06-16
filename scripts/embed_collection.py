#!/usr/bin/env python3
"""Generalized CLIP embedder — reads crawler manifest, writes vbaby flat index.

Replaces collection-specific embed_bayc.py with a manifest-driven pipeline:
  manifest.jsonl  (from nft_crawl.py export)
    → embeddings.f32 + meta.json + images/<tok>.jpg

Each manifest line:
  {"token_id": 123, "image_path": "...", "collection": "bayc", ...}

Row i in embeddings.f32 aligns with tokens[i] in meta.json — same contract
as embed_bayc.py / FlatIndex::open.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import open_clip
import torch
from PIL import Image


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.jsonl from nft_crawl export")
    ap.add_argument("--out", required=True, help="output dir for vbaby flat index")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    assert entries, "empty manifest"

    collection = entries[0].get("collection", "unknown")
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    device = pick_device(args.device)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.eval().to(device)
    print(f"loaded {args.model}/{args.pretrained} on {device}")

    tokens, all_feats = [], []
    batch_imgs, batch_toks = [], []
    d = None

    def flush():
        nonlocal batch_imgs, batch_toks, d
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        f = f.cpu().numpy().astype(np.float32)
        all_feats.append(f)
        tokens.extend(batch_toks)
        d = f.shape[1]
        batch_imgs, batch_toks = [], []
        print(f"  embedded {len(tokens)} ...", flush=True)

    for entry in entries:
        tok = int(entry["token_id"])
        img_path = entry["image_path"]
        img = Image.open(img_path).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((args.thumb, args.thumb))
        thumb.save(os.path.join(args.out, "images", f"{tok}.jpg"), "JPEG", quality=85)
        batch_imgs.append(preprocess(img))
        batch_toks.append(tok)
        if len(batch_imgs) >= args.batch:
            flush()
    flush()

    feats = np.concatenate(all_feats, axis=0)
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    with open(os.path.join(args.out, "embeddings.f32"), "wb") as fh:
        fh.write(feats.tobytes())

    meta = {
        "n": int(feats.shape[0]),
        "d": int(d),
        "model": f"{args.model}/{args.pretrained}",
        "tokens": tokens,
        "collection": collection,
    }
    # Include contract/chain from manifest if present
    if entries[0].get("contract"):
        meta["contract"] = entries[0]["contract"]
    if entries[0].get("chain"):
        meta["chain"] = entries[0]["chain"]

    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"wrote {feats.shape[0]} x {d} embeddings to {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
