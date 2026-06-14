#!/usr/bin/env python3
"""Embed the BAYC collection with CLIP and emit artifacts for vector-baby.

Outputs (under --out):
  embeddings.f32  : n*d little-endian float32, L2-normalized (row i <-> tokens[i])
  meta.json       : {"n", "d", "model", "tokens": [...]}
  images/<tok>.jpg: downscaled thumbnails for the web UI
"""
import argparse, glob, io, json, os, struct, time
import numpy as np
import torch, open_clip
from PIL import Image
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="data/bayc")
    ap.add_argument("--out", default="data/bayc")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--no-thumbs", action="store_true", help="skip writing thumbnails (already saved)")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.eval()
    print(f"loaded {args.model}/{args.pretrained} in {time.time()-t0:.1f}s")

    files = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    assert files, "no parquet files found"

    tokens, all_feats = [], []
    batch_imgs, batch_toks = [], []
    n_done = 0
    d = None

    def flush():
        nonlocal batch_imgs, batch_toks, n_done, d
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        f = f.cpu().numpy().astype(np.float32)
        all_feats.append(f)
        tokens.extend(batch_toks)
        n_done += len(batch_toks)
        d = f.shape[1]
        batch_imgs, batch_toks = [], []
        print(f"  embedded {n_done} ... ({time.time()-t0:.0f}s)", flush=True)

    for pf in files:
        tbl = pq.read_table(pf)
        for r in tbl.to_pylist():
            tok = int(r["token_metadata"].rstrip("/").split("/")[-1])
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            if not args.no_thumbs:
                thumb = img.copy()
                thumb.thumbnail((args.thumb, args.thumb))
                thumb.save(os.path.join(args.out, "images", f"{tok}.jpg"), "JPEG", quality=85)
            batch_imgs.append(preprocess(img))
            batch_toks.append(tok)
            if len(batch_imgs) >= args.batch:
                flush()
    flush()

    feats = np.concatenate(all_feats, axis=0)
    assert feats.shape[0] == len(tokens)
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    with open(os.path.join(args.out, "embeddings.f32"), "wb") as fh:
        fh.write(feats.tobytes())
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump({"n": int(feats.shape[0]), "d": int(d),
                   "model": f"{args.model}/{args.pretrained}",
                   "tokens": tokens}, fh)
    print(f"wrote {feats.shape[0]} x {d} embeddings in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
