#!/usr/bin/env python3
"""NFT Crawler CLI — Google-style NFT discovery and fetch pipeline.

Subcommands:
  init       Create data dirs and seed default collections
  seed       Register collections and enqueue tokens
  crawl      Run one or more crawl cycles
  status     Show crawl DB stats
  export     Export crawled images to embed-ready layout
  discover   List discoverable HF collections

Examples:
  python scripts/nft_crawl.py init
  python scripts/nft_crawl.py seed --collection bayc --limit 100
  python scripts/nft_crawl.py crawl --until-empty
  python scripts/nft_crawl.py export --collection bayc --out data/export/bayc
  python scripts/embed_collection.py --manifest data/export/bayc/manifest.jsonl --out data/bayc
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as script from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nft_crawler.config import CrawlerConfig
from nft_crawler.crawler import Crawler
from nft_crawler.discovery import (
    DEFAULT_SEEDS,
    discover_huggingface_collections,
    load_seeds_file,
    register_seeds,
    seed_to_collection,
)
from nft_crawler.export import export_collection
from nft_crawler.models import QueuePriority


def cmd_init(args: argparse.Namespace) -> None:
    config = CrawlerConfig(data_dir=args.data_dir)
    os.makedirs(config.data_dir, exist_ok=True)
    crawler = Crawler(config)
    seeds = load_seeds_file(args.seeds) if args.seeds else [seed_to_collection(s) for s in DEFAULT_SEEDS]
    n = register_seeds(crawler.db, seeds)
    print(f"initialized {config.data_dir}")
    print(f"registered {n} new collections ({len(seeds)} seeds total)")
    print(f"database: {config.db_path}")
    crawler.close()


def cmd_seed(args: argparse.Namespace) -> None:
    config = CrawlerConfig(data_dir=args.data_dir)
    crawler = Crawler(config)

    if args.collection:
        col = crawler.db.get_collection(args.collection)
        if not col:
            # try default seeds
            for s in DEFAULT_SEEDS:
                if s["slug"] == args.collection:
                    col = seed_to_collection(s)
                    crawler.db.upsert_collection(col)
                    break
        if not col:
            print(f"unknown collection: {args.collection}", file=sys.stderr)
            sys.exit(1)
        n = crawler.seed_collection(col, limit=args.limit, priority=QueuePriority.HIGH)
        print(f"seeded {n} new tokens for {args.collection} (limit={args.limit})")
    else:
        cols = crawler.db.list_collections()
        total = 0
        for col in cols:
            n = crawler.seed_collection(col, limit=args.limit)
            total += n
            print(f"  {col.slug}: {n} new tokens")
        print(f"total new tokens enqueued: {total}")
    crawler.close()


def cmd_crawl(args: argparse.Namespace) -> None:
    config = CrawlerConfig(data_dir=args.data_dir)
    crawler = Crawler(config)
    if args.until_empty:
        crawler.run_until_empty(
            batch_metadata=args.batch_metadata,
            batch_images=args.batch_images,
        )
    else:
        stats = crawler.run(
            max_metadata=args.batch_metadata,
            max_images=args.batch_images,
        )
        print(json.dumps(stats, indent=2))
    crawler.close()


def cmd_status(args: argparse.Namespace) -> None:
    config = CrawlerConfig(data_dir=args.data_dir)
    crawler = Crawler(config)
    stats = crawler.db.stats()
    print(json.dumps(stats, indent=2, default=str))
    if args.collection:
        by_status = crawler.db.count_tokens_by_status(args.collection)
        print(f"\n{args.collection}:", json.dumps(by_status, indent=2))
    crawler.close()


def cmd_export(args: argparse.Namespace) -> None:
    config = CrawlerConfig(data_dir=args.data_dir)
    crawler = Crawler(config)
    result = export_collection(crawler.db, args.collection, args.out)
    print(json.dumps(result, indent=2))
    crawler.close()


def cmd_discover(args: argparse.Namespace) -> None:
    hf = discover_huggingface_collections()
    print(f"Found {len(hf)} HuggingFace NFT datasets:")
    for col in hf:
        print(f"  {col.slug}: {col.hf_repo}")


def cmd_repair(args: argparse.Namespace) -> None:
    config = CrawlerConfig(data_dir=args.data_dir)
    crawler = Crawler(config)
    n = crawler.db.repair_frontier(args.collection)
    print(f"re-enqueued {n} stuck tokens")
    crawler.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="NFT crawler — discover and fetch NFT metadata/images")
    ap.add_argument("--data-dir", default="data/crawl", help="crawl data root")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="initialize DB and register seed collections")
    p_init.add_argument("--seeds", help="path to custom seeds JSON file")
    p_init.set_defaults(func=cmd_init)

    p_seed = sub.add_parser("seed", help="enqueue token IDs for crawling")
    p_seed.add_argument("--collection", help="single collection slug (default: all)")
    p_seed.add_argument("--limit", type=int, help="max tokens per collection")
    p_seed.set_defaults(func=cmd_seed)

    p_crawl = sub.add_parser("crawl", help="run crawl cycle(s)")
    p_crawl.add_argument("--batch-metadata", type=int, default=200)
    p_crawl.add_argument("--batch-images", type=int, default=200)
    p_crawl.add_argument("--until-empty", action="store_true", help="loop until frontier empty")
    p_crawl.set_defaults(func=cmd_crawl)

    p_status = sub.add_parser("status", help="show crawl statistics")
    p_status.add_argument("--collection")
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="export to embed-ready layout")
    p_export.add_argument("--collection", required=True)
    p_export.add_argument("--out", required=True)
    p_export.set_defaults(func=cmd_export)

    p_disc = sub.add_parser("discover", help="list discoverable collections")
    p_disc.set_defaults(func=cmd_discover)

    p_repair = sub.add_parser("repair", help="re-enqueue stuck tokens")
    p_repair.add_argument("--collection")
    p_repair.set_defaults(func=cmd_repair)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
