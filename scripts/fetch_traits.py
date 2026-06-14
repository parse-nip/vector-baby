#!/usr/bin/env python3
"""Fetch BAYC trait metadata from IPFS for ground-truth evaluation."""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

CID = "QmeSjSinHpPnmXmspMjwiXyN6zS4E9zccariGR3jxcaWtq"
GATEWAYS = [
    "https://gateway.pinata.cloud/ipfs/{cid}/{tok}",
    "https://ipfs.io/ipfs/{cid}/{tok}",
]


def fetch(tok):
    for g in GATEWAYS:
        try:
            url = g.format(cid=CID, tok=tok)
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.loads(r.read())
            attrs = {a["trait_type"]: a["value"] for a in d.get("attributes", [])}
            return tok, attrs
        except Exception:
            continue
    return tok, None


def main():
    toks = list(range(10000))
    out = {}
    fails = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (tok, attrs) in enumerate(ex.map(fetch, toks)):
            if attrs is None:
                fails.append(tok)
            else:
                out[str(tok)] = attrs
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/10000 ({len(fails)} fails)", flush=True)
    # retry fails once
    if fails:
        print(f"retrying {len(fails)} fails...", flush=True)
        with ThreadPoolExecutor(max_workers=12) as ex:
            for tok, attrs in ex.map(fetch, list(fails)):
                if attrs is not None:
                    out[str(tok)] = attrs
    with open("data/bayc/traits.json", "w") as f:
        json.dump(out, f)
    print(f"saved {len(out)} traits")
    furs = {}
    for a in out.values():
        furs[a.get("Fur", "?")] = furs.get(a.get("Fur", "?"), 0) + 1
    print("fur distribution:", json.dumps(furs, indent=0))


if __name__ == "__main__":
    main()
