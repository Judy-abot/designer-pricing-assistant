"""
Generates a synthetic handmade-clothing listings dataset with the exact same
columns as src/etsy_collector.py produces.

WHY THIS EXISTS
---------------
This sandbox can't reach api.etsy.com (outbound network is allowlisted to a
fixed set of dev domains), so there's no way to pull real listings from here.
This script fabricates a realistic stand-in dataset -- same schema, plausible
price/material/region/demand relationships -- so the feature engineering,
model training, and Streamlit app can all be built and tested end to end
right now. Swap it for data/raw/listings.csv (from etsy_collector.py, run on
a machine with real internet access) and everything downstream keeps working
unchanged.

Run:
    python src/generate_sample_data.py --n 3500 --out data/sample/listings_sample.csv
"""

import argparse
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RNG_SEED = 42

CATEGORIES = {
    # category: (base_price, category_path, taxonomy_id)
    "Dresses": (68, "Clothing|Women's Clothing|Dresses", 1201),
    "Tops & Blouses": (38, "Clothing|Women's Clothing|Tops", 1202),
    "Skirts": (42, "Clothing|Women's Clothing|Skirts", 1203),
    "Outerwear & Coats": (135, "Clothing|Outerwear|Coats", 1204),
    "Knitwear & Sweaters": (72, "Clothing|Knitwear|Sweaters", 1205),
    "Pants & Trousers": (58, "Clothing|Women's Clothing|Pants", 1206),
    "Men's Shirts": (52, "Clothing|Men's Clothing|Shirts", 1207),
}

# material: (price multiplier, weight in random draw)
MATERIALS = {
    "organic cotton": (1.05, 12),
    "cotton": (0.90, 14),
    "linen": (1.15, 10),
    "wool": (1.35, 8),
    "merino wool": (1.55, 5),
    "silk": (1.70, 6),
    "denim": (1.00, 8),
    "leather": (2.10, 4),
    "hemp": (1.10, 6),
    "recycled polyester": (0.85, 7),
    "cashmere": (2.60, 3),
    "velvet": (1.60, 4),
    "corduroy": (1.05, 6),
    "bamboo fabric": (1.20, 4),
}

# region: (cost multiplier, weight in random draw)
REGIONS = {
    "United States": (1.15, 30),
    "United Kingdom": (1.10, 14),
    "Canada": (1.05, 10),
    "France": (1.20, 8),
    "Germany": (1.12, 8),
    "Italy": (1.25, 6),
    "Netherlands": (1.08, 5),
    "Spain": (1.00, 5),
    "India": (0.55, 9),
    "Australia": (1.10, 5),
}

WHO_MADE = ["i_did", "collective"]
WHEN_MADE = ["made_to_order", "2020_2026", "2010_2019"]

ADJECTIVES = ["Handmade", "Boho", "Vintage-Style", "Minimalist", "Artisan", "Custom",
              "Sustainable", "Cottagecore", "Modern", "Classic"]


def weighted_choice(rng: np.random.Generator, keys: list, weights: list) -> str:
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return rng.choice(keys, p=w)


# Even weighting across categories (swap in real counts once you have them)
CATEGORY_KEYS = list(CATEGORIES.keys())
CATEGORY_WEIGHTS = [1.0] * len(CATEGORY_KEYS)


def generate(n: int, out_path: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    random.seed(RNG_SEED)

    rows = []
    for i in range(n):
        category = weighted_choice(rng, CATEGORY_KEYS, CATEGORY_WEIGHTS)
        base_price, category_path, taxonomy_id = CATEGORIES[category]

        n_materials = rng.integers(1, 3)
        material_keys = rng.choice(
            list(MATERIALS.keys()),
            size=n_materials,
            replace=False,
            p=np.array([v[1] for v in MATERIALS.values()]) / sum(v[1] for v in MATERIALS.values()),
        )
        material_mult = float(np.mean([MATERIALS[m][0] for m in material_keys]))

        region = weighted_choice(rng, list(REGIONS.keys()), [v[1] for v in REGIONS.values()])
        region_mult = REGIONS[region][0]

        # Latent "desirability" drives both favorites and a bit of price premium
        # (well-loved shops can command slightly higher prices) -- gives the
        # regression model a genuine, learnable demand signal.
        desirability = rng.normal(0, 1)

        price = (
            base_price
            * material_mult
            * region_mult
            * (1 + 0.05 * desirability)
            * rng.normal(1.0, 0.12)  # idiosyncratic noise
        )
        price = max(8.0, round(price, 2))

        views = max(1, int(rng.negative_binomial(6, 0.02) * (1 + 0.3 * desirability)))
        num_favorers = max(0, int(views * rng.uniform(0.03, 0.12) * (1 + 0.4 * max(desirability, -0.9))))

        n_tags = rng.integers(5, 13)
        tag_pool = [category.lower()] + [m for m in material_keys] + [region.lower(), "handmade", "gift"]
        tags = list(dict.fromkeys(tag_pool))[:n_tags]

        title = f"{rng.choice(ADJECTIVES)} {' '.join(material_keys)} {category}".title()

        rows.append({
            "listing_id": 1_000_000 + i,
            "title": title,
            "price": price,
            "currency_code": "USD",
            "quantity": int(rng.integers(1, 6)),
            "materials": "|".join(material_keys),
            "tags": "|".join(tags),
            "category_path": category_path,
            "taxonomy_id": taxonomy_id,
            "shop_id": 500_000 + int(rng.integers(0, n // 3)),  # shops list multiple items
            "region": region,
            "num_favorers": num_favorers,
            "views": views,
            "who_made": random.choice(WHO_MADE),
            "when_made": random.choice(WHEN_MADE),
            "url": f"https://www.etsy.com/listing/{1_000_000 + i}",
            "image_url": f"https://picsum.photos/seed/{1_000_000 + i}/400/400",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic listings to {out_path}")
    print(df[["category_path", "region", "price"]].groupby(["category_path"]).price.describe()[["mean", "min", "max"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Etsy-shaped listings data.")
    parser.add_argument("--n", type=int, default=3500, help="Number of rows to generate")
    parser.add_argument("--out", type=str, default="data/sample/listings_sample.csv", help="Output CSV path")
    args = parser.parse_args()
    generate(args.n, args.out)
