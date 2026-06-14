#!/usr/bin/env python3
"""Find the most 'solid gold' apes by pixel color (local ground truth, no net).
Renders a montage of the goldest apes."""
import glob, os, numpy as np
from PIL import Image, ImageDraw

files = glob.glob("data/bayc/images/*.jpg")
scores = []
for fp in files:
    tok = int(os.path.basename(fp).split(".")[0])
    im = np.asarray(Image.open(fp).convert("RGB").resize((64, 64))).astype(np.float32)
    # tight face patch (snout/cheeks); ape always fills the center
    c = im[30:52, 24:42]
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    # metallic gold = yellow-gold: R ~= G (NOT orange where R>>G), low blue.
    gold = (r > 170) & (g > 140) & (b < 110) & (np.abs(r - g) < 45) & ((g - b) > 70)
    frac = gold.mean()
    scores.append((frac, tok))
scores.sort(reverse=True)
top = scores[:16]
print("goldest apes:", ", ".join(f"#{t}({f:.2f})" for f, t in top))

cols, rows, cell = 4, 4, 200
mont = Image.new("RGB", (cols * cell, rows * cell), (15, 18, 24))
d = ImageDraw.Draw(mont)
for i, (f, t) in enumerate(top):
    rr, cc = i // cols, i % cols
    im = Image.open(f"data/bayc/images/{t}.jpg").convert("RGB").resize((cell - 8, cell - 28))
    mont.paste(im, (cc * cell + 4, rr * cell + 4))
    d.text((cc * cell + 6, rr * cell + cell - 22), f"#{t} {f:.2f}", fill=(200, 255, 200))
mont.save("/tmp/goldest.png")
print("saved /tmp/goldest.png")

import json
json.dump([t for _, t in scores[:60]], open("/tmp/gold_tokens.json", "w"))
