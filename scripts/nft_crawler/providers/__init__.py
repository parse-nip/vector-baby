"""Metadata providers — pluggable strategies per collection type."""

from __future__ import annotations

import glob
import io
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..models import Collection, FetchResult, MetadataSource, TokenRecord
from ..parser import extract_fields, parse_metadata_json
from ..resolver import metadata_uri_for_token, parse_data_uri, resolve_uri

if TYPE_CHECKING:
    from ..config import CrawlerConfig
    from ..fetcher import Fetcher


class MetadataProvider(ABC):
    @abstractmethod
    def fetch_metadata(
        self, collection: Collection, token_id: int, fetcher: Fetcher, config: CrawlerConfig
    ) -> tuple[bytes | None, str | None, str | None]:
        """Returns (json_bytes, resolved_uri, error)."""


class IpfsPatternProvider(MetadataProvider):
    """BAYC-style: metadata at ipfs://{cid}/{token_id}."""

    def fetch_metadata(self, collection, token_id, fetcher, config):
        uri = metadata_uri_for_token(collection, token_id)
        if not uri:
            return None, None, "no ipfs_cid configured"
        inline = parse_data_uri(uri)
        if inline:
            return inline, uri, None
        urls = resolve_uri(uri, config)
        result = fetcher.fetch_resolved(urls)
        if result.ok:
            return result.data, result.final_url, None
        return None, uri, result.error


class TokenUriProvider(MetadataProvider):
    """On-chain tokenURI — requires Alchemy or similar RPC (future).

    For now, tries common OpenSea metadata URL patterns as fallback.
    """

    OPENSEA_METADATA = "https://metadata.ens.domains/mainnet/avatar/{contract}/{token_id}"

    def fetch_metadata(self, collection, token_id, fetcher, config):
        # Try Alchemy NFT API if key present
        if config.alchemy_api_key:
            url = (
                f"https://eth-mainnet.g.alchemy.com/nft/v3/{config.alchemy_api_key}"
                f"/getNFTMetadata?contractAddress={collection.contract}"
                f"&tokenId={token_id}&refreshCache=false"
            )
            result = fetcher.fetch(url)
            if result.ok and result.data:
                return result.data, url, None

        # Reservoir API fallback
        if config.reservoir_api_key:
            url = (
                f"https://api.reservoir.tools/tokens/v7"
                f"?tokens={collection.contract}:{token_id}"
            )
            req_headers = {"x-api-key": config.reservoir_api_key}
            import urllib.request

            fetcher.politeness.wait(url)
            req = urllib.request.Request(url, headers={**req_headers, "User-Agent": config.user_agent})
            try:
                with urllib.request.urlopen(req, timeout=config.request_timeout) as resp:
                    data = resp.read()
                    return data, url, None
            except Exception as e:
                pass

        return None, None, "token_uri requires ALCHEMY_API_KEY or RESERVOIR_API_KEY"


class HuggingFaceProvider(MetadataProvider):
    """Read images/metadata from local HF parquet (after download)."""

    def fetch_metadata(self, collection, token_id, fetcher, config):
        if not collection.hf_repo:
            return None, None, "no hf_repo"
        parquet_dir = os.path.join(config.data_dir, "parquet", collection.slug)
        files = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
        if not files:
            return None, None, f"no parquet in {parquet_dir} (run download first)"

        import pyarrow.parquet as pq

        for pf in files:
            tbl = pq.read_table(pf)
            for r in tbl.to_pylist():
                tok = int(r["token_metadata"].rstrip("/").split("/")[-1])
                if tok == token_id:
                    # Synthesize minimal metadata JSON from parquet row
                    import json

                    meta = {
                        "name": f"#{token_id}",
                        "image": f"hf-parquet://{collection.slug}/{token_id}",
                        "attributes": [],
                    }
                    return json.dumps(meta).encode(), f"hf://{collection.slug}/{token_id}", None
        return None, None, f"token {token_id} not in parquet"


def get_provider(source: MetadataSource) -> MetadataProvider:
    return {
        MetadataSource.IPFS_PATTERN: IpfsPatternProvider(),
        MetadataSource.TOKEN_URI: TokenUriProvider(),
        MetadataSource.HUGGINGFACE: HuggingFaceProvider(),
        MetadataSource.HTTP_API: TokenUriProvider(),
    }[source]


def fetch_token_image(
    image_uri: str,
    fetcher: Fetcher,
    config: CrawlerConfig,
) -> FetchResult:
    inline = parse_data_uri(image_uri)
    if inline:
        return FetchResult(ok=True, data=inline, content_type="application/octet-stream")
    if image_uri.startswith("hf-parquet://"):
        return _fetch_hf_image(image_uri, config)
    urls = resolve_uri(image_uri, config)
    return fetcher.fetch_resolved(urls)


def _fetch_hf_image(uri: str, config: CrawlerConfig) -> FetchResult:
    # hf-parquet://bayc/123
    parts = uri.replace("hf-parquet://", "").split("/")
    slug, token_id = parts[0], int(parts[1])
    parquet_dir = os.path.join(config.data_dir, "parquet", slug)
    files = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
    import pyarrow.parquet as pq
    from PIL import Image

    for pf in files:
        tbl = pq.read_table(pf)
        for r in tbl.to_pylist():
            tok = int(r["token_metadata"].rstrip("/").split("/")[-1])
            if tok == token_id:
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                return FetchResult(ok=True, data=buf.getvalue(), content_type="image/jpeg")
    return FetchResult(ok=False, error=f"image not found in parquet for {uri}")
