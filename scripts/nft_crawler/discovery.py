"""Collection discovery — seed registry and expansion.

Google discovers new pages by following links. We discover new collections via:
  1. Seed file (curated list of known high-value collections)
  2. HuggingFace NFT datasets (huggingnft/* repos)
  3. (Future) Reservoir/OpenSea collection APIs when API keys present

Seeds are the "sitemap.xml" equivalent — known entry points we trust.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from .models import Chain, Collection, MetadataSource

if TYPE_CHECKING:
    from .db import CrawlDB

# Curated seeds — expand over time. Each entry is a known-good collection
# with a documented metadata resolution strategy.
DEFAULT_SEEDS: list[dict] = [
    {
        "slug": "bayc",
        "name": "Bored Ape Yacht Club",
        "chain": "ethereum",
        "contract": "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",
        "supply": 10000,
        "metadata_source": "ipfs_pattern",
        "ipfs_cid": "QmeSjSinHpPnmXmspMjwiXyN6zS4E9zccariGR3jxcaWtq",
        "opensea_slug": "boredapeyachtclub",
        "hf_repo": "huggingnft/boredapeyachtclub",
    },
]


def seed_to_collection(seed: dict) -> Collection:
    return Collection(
        slug=seed["slug"],
        name=seed["name"],
        chain=Chain(seed["chain"]),
        contract=seed["contract"],
        supply=seed.get("supply"),
        metadata_source=MetadataSource(seed["metadata_source"]),
        ipfs_cid=seed.get("ipfs_cid"),
        hf_repo=seed.get("hf_repo"),
        opensea_slug=seed.get("opensea_slug"),
        extra=seed.get("extra", {}),
    )


def load_seeds_file(path: str) -> list[Collection]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [seed_to_collection(s) for s in data]
    return [seed_to_collection(s) for s in data.get("collections", [])]


def register_seeds(db: CrawlDB, seeds: list[Collection]) -> int:
    """Insert seed collections. Returns count of new registrations."""
    existing = {c.slug for c in db.list_collections()}
    new = 0
    for col in seeds:
        if col.slug not in existing:
            new += 1
        db.upsert_collection(col)
    return new


def enumerate_token_ids(collection: Collection, limit: int | None = None) -> list[int]:
    """Generate token IDs to crawl for a collection."""
    if collection.supply is not None:
        ids = list(range(collection.supply))
    else:
        # Unknown supply — caller must discover via events/API
        raise ValueError(
            f"collection {collection.slug} has no supply; use provider to discover tokens"
        )
    if limit is not None:
        ids = ids[:limit]
    return ids


def discover_huggingface_collections() -> list[Collection]:
    """List huggingnft/* datasets as discoverable collections."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    api = HfApi()
    cols: list[Collection] = []
    try:
        datasets = api.list_datasets(author="huggingnft", limit=50)
        for ds in datasets:
            repo = ds.id
            slug = repo.split("/")[-1].replace("-", "_")[:32]
            cols.append(
                Collection(
                    slug=slug,
                    name=slug.replace("_", " ").title(),
                    chain=Chain.ETHEREUM,
                    contract="0x0000000000000000000000000000000000000000",
                    metadata_source=MetadataSource.HUGGINGFACE,
                    hf_repo=repo,
                    extra={"hf_only": True},
                )
            )
    except Exception:
        pass
    return cols
