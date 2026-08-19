"""
Etsy Open API v3 collector for handmade-clothing listings.

WHERE TO RUN THIS
------------------
This script talks to https://api.etsy.com, which is NOT reachable from
sandboxed dev environments that restrict outbound
network access to an allowlist. Run this on your own machine, a CI runner,
or any host with normal internet access.

SETUP
-----
1. Register a free app at https://www.etsy.com/developers/register
   (approval is usually instant for the "Personal" access level).
2. On Your Apps, you'll see both a "Keystring" and a "Shared secret" --
   you need both, not just the keystring.
3. Export both as environment variables (never hardcode them):
       export ETSY_API_KEY="your_keystring_here"
       export ETSY_SHARED_SECRET="your_shared_secret_here"
4. pip install -r requirements.txt
5. Run:
       python src/etsy_collector.py --target 3000 --out data/raw/listings.csv

AUTH NOTES
----------
Every request needs an `x-api-key` header formatted as
`<keystring>:<shared_secret>` (colon-joined) -- this is required on every
v3 endpoint, regardless of whether it's public or scoped. The
findAllListingsActive / getBuyerTaxonomyNodes endpoints used here return
public marketplace data and don't additionally need an OAuth user token
-- OAuth is only required for endpoints that touch a specific shop's
private data, e.g. creating or editing a listing.

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
from typing import Optional

import requests

API_BASE = "https://api.etsy.com/v3/application"
REQUEST_SLEEP_SECONDS = 0.4  # ~2.5 req/s -- eased off from 0.25s after shop
                              # lookups (roughly doubling total request
                              # volume) started triggering 429s in practice
MAX_RETRIES = 5

# Keywords used to auto-discover relevant taxonomy nodes under the
# clothing tree. Edit this list to widen/narrow what gets pulled.
CLOTHING_KEYWORDS = [
    "women's clothing",
    "men's clothing",
    "dresses",
    "tops",
    "skirts",
    "sweaters",
    "pullover",
    "knit",
    "cardigan",
    # "outerwear" and "knitwear" (the original terms here) never actually
    # matched anything -- Etsy's real category names say "Coats" and
    # "Jackets", not the generic umbrella word. Confirmed by inspecting a
    # real 40-node result: zero outerwear/pants categories were found at
    # all with the old list.
    "coat",
    "jacket",
    "blazer",
    "parka",
    "pants",
    "trousers",
    "leggings",
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
    def __init__(self, api_key: str, shared_secret: str):
        if not api_key or not shared_secret:
            raise ValueError(
                "Missing Etsy credentials. Set both the ETSY_API_KEY (keystring) "
                "and ETSY_SHARED_SECRET environment variables -- both are required "
                "to build the x-api-key header."
            )
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": f"{api_key}:{shared_secret}"})

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
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

    def get_shop(self, shop_id: int) -> dict:
        return self._get(f"/shops/{shop_id}")

    def get_listing_images(self, listing_id: int) -> dict:
        return self._get(f"/listings/{listing_id}/images")

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


def find_clothing_root(nodes: list) -> Optional[dict]:
    """Finds the top-level 'Clothing' taxonomy node so we only search its subtree
    (rather than Etsy's entire category tree, which also covers toys, home
    decor, pet supplies, etc. -- several of which have subcategories that
    happen to share words with clothing terms, e.g. "Yo-Yos & Tops" (toys)
    or "Tree Skirts" (holiday decor))."""
    for node in nodes:
        if (node.get("name") or "").strip().lower() == "clothing":
            return node
    return None


def find_subcategory_root(parent_node: dict, target_name: str) -> Optional[dict]:
    """Finds a node matching target_name (case-insensitive) anywhere within
    parent_node's subtree. Used to scope the search to a specific branch,
    e.g. 'Women's Clothing' within the broader 'Clothing' tree -- this is
    also what separates the ~9 near-duplicate 'Sweaters' nodes we found
    earlier (one per gender/age branch) into a single, correctly-scoped set."""
    for child in parent_node.get("children") or []:
        if (child.get("name") or "").strip().lower() == target_name.lower():
            return child
        found = find_subcategory_root(child, target_name)
        if found:
            return found
    return None


def find_clothing_taxonomy_ids(client: EtsyClient, scope_to: Optional[str] = None) -> dict[int, str]:
    print("Discovering clothing taxonomy nodes...")
    tree = client.get_buyer_taxonomy_nodes()

    clothing_root = find_clothing_root(tree)
    if clothing_root is None:
        print("  Could not find a top-level 'Clothing' node -- falling back to a "
              "full-tree search (results may include false positives from "
              "unrelated categories, e.g. toys or home decor).")
        search_space = flatten_taxonomy(tree)
    else:
        print(f"  Found top-level 'Clothing' node (id={clothing_root['id']}) -- searching only within it.")
        search_root = clothing_root
        if scope_to:
            scoped_node = find_subcategory_root(clothing_root, scope_to)
            if scoped_node is not None:
                print(f"  Found '{scope_to}' node (id={scoped_node['id']}) -- scoping search to it only.")
                search_root = scoped_node
            else:
                print(f"  Could not find a '{scope_to}' node under Clothing -- "
                      f"falling back to the full Clothing tree.")
        search_space = flatten_taxonomy(search_root.get("children") or [])

    matches = {}
    for node in search_space:
        name = (node.get("name") or "").lower()
        if any(kw in name for kw in CLOTHING_KEYWORDS):
            matches[node["id"]] = node.get("name")

    print(f"  Found {len(matches)} matching taxonomy nodes: {list(matches.values())}")
    return matches


# Canonical garment-type buckets. Many of Etsy's real taxonomy nodes are
# near-duplicates of each other at different tree depths (e.g. "Sweaters"
# shows up under separate Women's/Men's/Kids' branches as 9 distinct node
# ids that all mean the same garment type -- confirmed on a real pull).
# Splitting the collection target evenly across every raw node id gives
# an uneven, overly fragmented spread. Bucketing by canonical type first,
# then splitting the target across BUCKETS, gives each real garment type
# a fair, deep share regardless of how many duplicate node ids represent it.
GARMENT_BUCKETS = {
    "Dresses": ["dress"],
    "Tops": ["top", "blouse", "tee", "halter"],
    "Skirts": ["skirt", "kilt"],
    "Sweaters & Knitwear": ["sweater", "knit", "cardigan", "pullover"],
    "Outerwear & Coats": ["coat", "jacket", "outerwear", "parka", "blazer"],
    "Pants & Trousers": ["pant", "trouser", "legging"],
}
# Overly broad umbrella nodes that duplicate their own children -- pulling
# from these separately would just re-fetch a mixed bag of whatever's
# already covered by the specific buckets above.
EXCLUDED_NODE_NAMES = {"men's clothing", "women's clothing", "clothing"}


def bucket_taxonomy_ids(taxonomy_ids: dict) -> dict:
    """Groups {taxonomy_id: name} into {bucket_label: [taxonomy_id, ...]},
    consolidating near-duplicate nodes into one canonical garment type."""
    buckets = {label: [] for label in GARMENT_BUCKETS}
    buckets["Other"] = []
    for tid, name in taxonomy_ids.items():
        name_lower = (name or "").lower()
        if name_lower in EXCLUDED_NODE_NAMES:
            continue
        placed = False
        for label, keywords in GARMENT_BUCKETS.items():
            if any(kw in name_lower for kw in keywords):
                buckets[label].append(tid)
                placed = True
                break
        if not placed:
            buckets["Other"].append(tid)
    # Drop empty buckets so the target split isn't diluted by categories
    # that genuinely have zero matching nodes this run.
    return {label: ids for label, ids in buckets.items() if ids}


def extract_row(listing: dict, category_name: str, region: str, image_url: Optional[str] = None) -> dict:
    price = listing.get("price", {}) or {}
    amount = price.get("amount")
    divisor = price.get("divisor") or 1
    price_value = round(amount / divisor, 2) if amount is not None else None

    return {
        "listing_id": listing.get("listing_id"),
        "title": listing.get("title"),
        "price": price_value,
        "currency_code": price.get("currency_code"),
        "quantity": listing.get("quantity"),
        "materials": "|".join(listing.get("materials") or []),
        "tags": "|".join(listing.get("tags") or []),
        # category_path isn't reliably present on the real v3 response --
        # rather than trust a field that may not exist, use the category
        # name we already know, since we specifically queried this
        # taxonomy_id to get this listing in the first place.
        "category_path": category_name,
        "taxonomy_id": listing.get("taxonomy_id"),
        "shop_id": listing.get("shop_id"),
        # region comes from a separate /shops/{shop_id} lookup, cached per
        # shop -- there's no embedded shop object on the listing itself
        # (confirmed by inspecting a raw response: no "shop" key at all).
        "region": region,
        "num_favorers": listing.get("num_favorers", 0),
        "views": listing.get("views", 0),
        "who_made": listing.get("who_made"),
        "when_made": listing.get("when_made"),
        "url": listing.get("url"),
        "image_url": image_url,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def collect(target: int, out_path: str, scope_to: Optional[str] = None, fetch_images: bool = False) -> None:
    api_key = os.environ.get("ETSY_API_KEY")
    shared_secret = os.environ.get("ETSY_SHARED_SECRET")
    client = EtsyClient(api_key, shared_secret)

    taxonomy_ids = find_clothing_taxonomy_ids(client, scope_to=scope_to)
    if not taxonomy_ids:
        print("No taxonomy nodes matched CLOTHING_KEYWORDS -- edit the list and retry.")
        sys.exit(1)

    buckets = bucket_taxonomy_ids(taxonomy_ids)
    if not buckets:
        print("No garment-type buckets matched -- edit GARMENT_BUCKETS and retry.")
        sys.exit(1)
    print(f"Consolidated {len(taxonomy_ids)} raw taxonomy nodes into "
          f"{len(buckets)} garment-type buckets: {list(buckets.keys())}")
    if fetch_images:
        print("  --fetch-images enabled: one extra API call per listing (not cached), "
              "expect roughly double the requests and runtime of a normal pull.")

    # Give each BUCKET (not each raw node) a fixed, even share of the
    # target. This is what actually fixes the thin-per-category problem --
    # e.g. "Sweaters" used to be split across 9 near-duplicate node ids
    # (each getting its own small slice); now the whole Sweaters bucket
    # gets one full, fair share, pulling from whichever of its member
    # node ids are needed to fill it.
    per_bucket_target = max(1, target // len(buckets))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen_ids = set()
    collected = 0
    shop_region_cache = {}  # shop_id -> region, avoids re-fetching the same shop repeatedly

    def get_region(shop_id) -> str:
        if shop_id not in shop_region_cache:
            try:
                shop = client.get_shop(shop_id)
                region = shop.get("shop_location_country_iso") or shop.get("shipping_from_country_iso") or ""
            except (requests.HTTPError, RuntimeError) as e:
                # A single shop being persistently rate-limited or erroring
                # shouldn't crash the whole run -- fall back to empty
                # region for this shop and keep going. engineer_features
                # already treats empty region as "Unknown" safely.
                print(f"  [warn] Couldn't fetch shop {shop_id} ({e}); region left blank for this shop")
                region = ""
            shop_region_cache[shop_id] = region
        return shop_region_cache[shop_id]

    def get_first_image_url(listing_id) -> Optional[str]:
        # Unlike shop/region, images aren't cacheable across listings --
        # every listing has its own set. Confirmed via GET
        # /listings/{id}/images that the endpoint returns {"results": [...
        # {"url_570xN": ..., "rank": 1, ...}]}, sorted by rank; take the
        # first (primary) image.
        try:
            data = client.get_listing_images(listing_id)
            results = data.get("results") or []
            if results:
                return results[0].get("url_570xN")
        except (requests.HTTPError, RuntimeError) as e:
            print(f"  [warn] Couldn't fetch images for listing {listing_id} ({e})")
        return None

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

        for bucket_label, tid_list in buckets.items():
            if collected >= target:
                break
            bucket_collected = 0
            print(f"Pulling bucket: {bucket_label} ({len(tid_list)} taxonomy nodes), "
                  f"up to {per_bucket_target} from this bucket")

            for taxonomy_id in tid_list:
                if collected >= target or bucket_collected >= per_bucket_target:
                    break
                offset = 0
                page_limit = 100
                while collected < target and bucket_collected < per_bucket_target:
                    try:
                        data = client.find_active_listings(
                            taxonomy_id, limit=page_limit, offset=offset
                        )
                    except requests.HTTPError as e:
                        print(f"  HTTP error on node {taxonomy_id}, moving to next: {e}")
                        break

                    results = data.get("results", [])
                    if not results:
                        break  # exhausted this node

                    for listing in results:
                        lid = str(listing.get("listing_id"))
                        if lid in seen_ids:
                            continue
                        seen_ids.add(lid)
                        # Only count designer-made items toward the target --
                        # resale/vintage/collective listings (who_made !=
                        # "i_did") get filtered out during training anyway,
                        # so checking here avoids spending API quota pulling
                        # listings that will never actually be used.
                        if listing.get("who_made") != "i_did":
                            continue
                        # Etsy flags craft supplies / digital patterns /
                        # add-ons with is_supply=True -- these aren't
                        # finished garments (found one contaminating the
                        # data: a "Design Elements" add-on sitting in a
                        # clothing taxonomy).
                        if listing.get("is_supply"):
                            continue
                        region = get_region(listing.get("shop_id"))
                        image_url = get_first_image_url(listing.get("listing_id")) if fetch_images else None
                        # Use the consolidated bucket label as the category,
                        # not the raw near-duplicate node name -- this also
                        # means "category" ends up with a handful of clean,
                        # meaningful values instead of 9 near-identical
                        # "Sweaters" variants.
                        row = extract_row(listing, bucket_label, region, image_url)
                        writer.writerow(row)
                        collected += 1
                        bucket_collected += 1
                        if collected >= target or bucket_collected >= per_bucket_target:
                            break

                    offset += page_limit
                    print(f"    ...{collected}/{target} total, "
                          f"{bucket_collected}/{per_bucket_target} this bucket")

    print(f"Done. Wrote {collected} listings to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect handmade-clothing listings from Etsy.")
    parser.add_argument("--target", type=int, default=3000, help="Total listings to collect")
    parser.add_argument("--out", type=str, default="data/raw/listings.csv", help="Output CSV path")
    parser.add_argument("--scope", type=str, default=None,
                         help="Optional taxonomy branch to scope the search to, e.g. \"Women's Clothing\". "
                              "Defaults to the whole Clothing tree if omitted.")
    parser.add_argument("--fetch-images", action="store_true",
                         help="Also fetch each listing's primary image URL. One extra API call per "
                              "listing (not cacheable, unlike shop/region lookups) -- roughly doubles "
                              "total requests and runtime. Not needed for Project 1; exists for "
                              "Project 4, which reuses this dataset and needs real images.")
    args = parser.parse_args()
    collect(args.target, args.out, scope_to=args.scope, fetch_images=args.fetch_images)
