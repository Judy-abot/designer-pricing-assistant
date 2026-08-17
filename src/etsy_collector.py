"""
Etsy Open API v3 collector for handmade-clothing listings.

WHERE TO RUN THIS
------------------
This script talks to https://api.etsy.com, which is NOT reachable from
sandboxed dev environments (Claude's included) that restrict outbound
network access to an allowlist. Run this on your own machine, a CI runner,
or any host with normal internet access.

SETUP
-----
1. Register a free app at https://www.etsy.com/developers/register
   (approval is usually instant for the "Personal" access level).
2. Copy the "Keystring" shown on Your Apps -> this is your API key.
3. Export it as an environment variable (never hardcode it):
       export ETSY_API_KEY="your_keystring_here"
4. pip install -r requirements.txt
5. Run:
       python src/etsy_collector.py --target 3000 --out data/raw/listings.csv

AUTH NOTES
----------
Every request needs the `x-api-key: <keystring>` header. The
findAllListingsActive / getBuyerTaxonomyNodes endpoints used here return
public marketplace data and only require the API key -- no OAuth user
token needed. (OAuth is only required for endpoints that touch a specific
shop's private data, e.g. creating or editing a listing.)

WHAT IT COLLECTS
----------------
For each listing: price, currency, materials, tags, category path,
taxonomy id, shop id, shop region (country), favorites (num_favorers,
used later as a demand proxy), views, listing url, and the first image
url. Materials, shop/collection id, and image urls are saved (not just
price) because Project 4 reuses this exact dataset.

RATE LIMITS
-----------
Standard keys are limited to ~10 requests/second and a daily quota
(check Your Apps for your exact quota). This script sleeps between calls
and retries with backoff on 429/5xx.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

API_BASE = "https://api.etsy.com/v3/application"
REQUEST_SLEEP_SECONDS = 0.25  # ~4 req/s, safely under the 10 req/s cap
MAX_RETRIES = 5

# Keywords used to auto-discover relevant taxonomy nodes under the
# clothing tree. Edit this list to widen/narrow what gets pulled.
CLOTHING_KEYWORDS = [
    "women's clothing",
    "men's clothing",
    "dresses",
    "tops",
    "skirts",
    "outerwear",
    "knitwear",
    "sweaters",
]

CSV_FIELDS = [
    "listing_id",
    "title",
    "price",
    "currency_code",
    "quantity",
    "materials",
    "tags",
    "category_path",
    "taxonomy_id",
    "shop_id",
    "region",
    "num_favorers",
    "views",
    "who_made",
    "when_made",
    "url",
    "image_url",
    "collected_at",
]


class EtsyClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "No Etsy API key found. Set the ETSY_API_KEY environment variable."
            )
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key})

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{API_BASE}{path}"
        for attempt in range(1, MAX_RETRIES + 1):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                time.sleep(REQUEST_SLEEP_SECONDS)
                return resp.json()
            if resp.status_code in (429, 500, 502, 503):
                wait = min(2 ** attempt, 30)
                print(f"  [retry] {resp.status_code} on {path}, backing off {wait}s")
                time.sleep(wait)
                continue
            # Non-retryable error
            resp.raise_for_status()
        raise RuntimeError(f"Exceeded retries for {url}")

    def get_buyer_taxonomy_nodes(self) -> list[dict]:
        data = self._get("/buyer-taxonomy/nodes")
        return data.get("results", [])

    def find_active_listings(
        self, taxonomy_id: int, limit: int = 100, offset: int = 0
    ) -> dict:
        params = {
            "taxonomy_id": taxonomy_id,
            "limit": limit,
            "offset": offset,
            "sort_on": "score",
            "sort_order": "desc",
            "includes": "Shop,Images",
        }
        return self._get("/listings/active", params=params)


def flatten_taxonomy(nodes: list[dict]) -> list[dict]:
    """Recursively flatten the taxonomy tree (nodes have a 'children' key)."""
    flat = []
    for node in nodes:
        flat.append(node)
        children = node.get("children") or []
        if children:
            flat.extend(flatten_taxonomy(children))
    return flat


def find_clothing_taxonomy_ids(client: EtsyClient) -> dict[int, str]:
    print("Discovering clothing taxonomy nodes...")
    tree = client.get_buyer_taxonomy_nodes()
    flat = flatten_taxonomy(tree)

    matches = {}
    for node in flat:
        name = (node.get("name") or "").lower()
        if any(kw in name for kw in CLOTHING_KEYWORDS):
            matches[node["id"]] = node.get("name")

    print(f"  Found {len(matches)} matching taxonomy nodes: {list(matches.values())}")
    return matches


def extract_row(listing: dict) -> dict:
    price = listing.get("price", {}) or {}
    amount = price.get("amount")
    divisor = price.get("divisor") or 1
    price_value = round(amount / divisor, 2) if amount is not None else None

    shop = listing.get("shop") or {}
    region = shop.get("shop_location") or shop.get("country_iso") or ""

    images = listing.get("images") or []
    image_url = images[0].get("url_570xN") if images else None

    return {
        "listing_id": listing.get("listing_id"),
        "title": listing.get("title"),
        "price": price_value,
        "currency_code": price.get("currency_code"),
        "quantity": listing.get("quantity"),
        "materials": "|".join(listing.get("materials") or []),
        "tags": "|".join(listing.get("tags") or []),
        "category_path": "|".join(listing.get("category_path") or []),
        "taxonomy_id": listing.get("taxonomy_id"),
        "shop_id": listing.get("shop_id"),
        "region": region,
        "num_favorers": listing.get("num_favorers", 0),
        "views": listing.get("views", 0),
        "who_made": listing.get("who_made"),
        "when_made": listing.get("when_made"),
        "url": listing.get("url"),
        "image_url": image_url,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def collect(target: int, out_path: str) -> None:
    api_key = os.environ.get("ETSY_API_KEY")
    client = EtsyClient(api_key)

    taxonomy_ids = find_clothing_taxonomy_ids(client)
    if not taxonomy_ids:
        print("No taxonomy nodes matched CLOTHING_KEYWORDS -- edit the list and retry.")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen_ids = set()
    collected = 0

    # Resume support: skip listing_ids already in an existing output file
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen_ids.add(row["listing_id"])
        collected = len(seen_ids)
        print(f"Resuming: {collected} listings already in {out_path}")

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for taxonomy_id, name in taxonomy_ids.items():
            if collected >= target:
                break
            print(f"Pulling category: {name} (taxonomy_id={taxonomy_id})")
            offset = 0
            page_limit = 100
            while collected < target:
                try:
                    data = client.find_active_listings(
                        taxonomy_id, limit=page_limit, offset=offset
                    )
                except requests.HTTPError as e:
                    print(f"  HTTP error, moving to next category: {e}")
                    break

                results = data.get("results", [])
                if not results:
                    break  # exhausted this category

                for listing in results:
                    lid = str(listing.get("listing_id"))
                    if lid in seen_ids:
                        continue
                    seen_ids.add(lid)
                    row = extract_row(listing)
                    writer.writerow(row)
                    collected += 1
                    if collected >= target:
                        break

                offset += page_limit
                print(f"    ...{collected}/{target} listings collected")

    print(f"Done. Wrote {collected} listings to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect handmade-clothing listings from Etsy.")
    parser.add_argument("--target", type=int, default=3000, help="Total listings to collect")
    parser.add_argument("--out", type=str, default="data/raw/listings.csv", help="Output CSV path")
    args = parser.parse_args()
    collect(args.target, args.out)
