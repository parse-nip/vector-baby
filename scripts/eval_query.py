#!/usr/bin/env python3
"""Iterate on text-query strategies and render a montage of the top results
(viewed directly by the agent for visual ground truth)."""
import argparse, json, os
import numpy as np
import torch, open_clip
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--q", required=True)
ap.add_argument("--baseline", default="", help="optional prompt to subtract (isolates the attribute direction)")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--out", default="/tmp/montage.png")
ap.add_argument("--model", default="ViT-L-14")
ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
args = ap.parse_args()

meta = json.load(open("data/bayc/meta.json"))
tokens = meta["tokens"]; d = meta["d"]
X = np.fromfile("data/bayc/embeddings.f32", dtype=np.float32).reshape(-1, d)

model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
model.eval()
tok = open_clip.get_tokenizer(args.model)

def emb(text):
    with torch.no_grad():
        f = model.encode_text(tok([text]))
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().numpy().astype(np.float32)

q = emb(args.q)
if args.baseline:
    q = q - args.alpha * emb(args.baseline)
    q = q / (np.linalg.norm(q) + 1e-9)

sims = X @ q
order = np.argsort(-sims)[:args.k]
res = [(tokens[i], float(sims[i])) for i in order]
print("query:", args.q, "| baseline:", args.baseline or "(none)")
print("top:", ", ".join(f"#{t}({s:.3f})" for t, s in res))

# montage
cols = 4; rows = (args.k + cols - 1) // cols
cell = 200
mont = Image.new("RGB", (cols * cell, rows * cell), (15, 18, 24))
draw = ImageDraw.Draw(mont)
for idx, (t, s) in enumerate(res):
    r, c = idx // cols, idx % cols
    fp = f"data/bayc/images/{t}.jpg"
    if os.path.isfile(fp):
        im = Image.open(fp).convert("RGB").resize((cell - 8, cell - 28))
        mont.paste(im, (c * cell + 4, r * cell + 4))
    draw.text((c * cell + 6, r * cell + cell - 22), f"#{t}  {s:.3f}", fill=(200, 255, 200))
mont.save(args.out)
print("saved", args.out)
