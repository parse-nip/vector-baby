"""Metadata JSON parser — extract image URLs, traits, names.

Follows OpenSea metadata standards (de-facto ERC-721 metadata schema):
  https://docs.opensea.io/docs/metadata-standards
"""

from __future__ import annotations

import json
from typing import Any


def parse_metadata_json(raw: bytes) -> dict[str, Any]:
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("metadata root must be object")
    return data


def extract_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Pull normalized fields from raw metadata JSON."""
    traits: dict[str, str] = {}

    # OpenSea attributes array
    for attr in data.get("attributes", []):
        if isinstance(attr, dict):
            t = attr.get("trait_type") or attr.get("traitType")
            v = attr.get("value")
            if t is not None and v is not None:
                traits[str(t)] = str(v)

    # properties.attributes (older CryptoPunks-style)
    props = data.get("properties", {})
    if isinstance(props, dict):
        for attr in props.get("attributes", []):
            if isinstance(attr, dict):
                t = attr.get("trait_type") or attr.get("traitType")
                v = attr.get("value")
                if t is not None and v is not None:
                    traits[str(t)] = str(v)

    image = data.get("image") or data.get("image_url") or data.get("imageUrl")
    animation = data.get("animation_url") or data.get("animationUrl")

    return {
        "name": data.get("name"),
        "description": data.get("description"),
        "image_uri": image,
        "animation_uri": animation,
        "traits": traits,
        "raw": data,
    }
