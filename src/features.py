"""
Feature engineering shared by train_model.py and app.py.

A DELIBERATE MODELING CHOICE ON "FAVORITES"
--------------------------------------------
The project brief lists favorites as an engineered *feature*. In practice,
a brand-new listing a designer is about to publish always has 0 favorites
and 0 views -- so if `num_favorers` were a model *input*, every single
prediction request would feed it the same constant (0), which wastes the
signal and can bias predictions (the model would have learned "0 favorites"
patterns from a training set where almost no real listing had exactly 0).

Instead, favorites/views are used as a *sample weight* during training:
listings the market has already validated (lots of favorites) count more
when the model learns price patterns, while still letting the model be
queried with just material/category/region/etc. for a listing that doesn't
have any engagement yet. This keeps "favorites as a demand proxy" in the
spirit of the brief while keeping the app usable for new, unlisted items.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["category", "primary_material", "region", "when_made"]
NUMERIC_COLS = ["material_count", "log_quantity"]

ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# Static USD conversion snapshot (approximate mid-market rates, Aug 2026).
# Real Etsy listings arrive priced in whatever currency the seller uses --
# without converting, a model sees "635000" (VND, ~$24) and "45" (USD, $45)
# as if they were on the same scale, which wrecks both accuracy and the
# error metrics. This is a fixed reference point, not a live feed -- good
# enough to make one training run internally consistent, but rates will
# have drifted if this gets reused much later; refresh the table then.
CURRENCY_TO_USD = {
    "USD": 1.0,
    "GBP": 1.34,
    "EUR": 1.156,
    "CAD": 0.722,
    "AUD": 0.70,
    "NZD": 0.585,
    "HKD": 0.128,
    "MYR": 0.25,
    "PHP": 0.0166,
    "IDR": 0.000059,
    "VND": 0.0000382,
    "TRY": 0.0211,
    "SEK": 0.107,
    "NOK": 0.106,
    "ILS": 0.334,
    # Added after the women's-scoping pull surfaced these -- INR alone
    # accounted for 378 of 400 dropped rows, the largest currency gap
    # found so far.
    "INR": 0.01048,
    "SGD": 0.7819,
    "MXN": 0.0588,
    "CHF": 1.230,
    "DKK": 0.1550,
    "MAD": 0.1053,  # approximate -- less liquid currency, wider spread than the others
}


def convert_to_usd(price: pd.Series, currency_code: pd.Series) -> pd.Series:
    """Converts a price column to USD. Rows with a currency not in
    CURRENCY_TO_USD become NaN, so they get dropped downstream rather than
    silently mispriced."""
    rates = currency_code.map(CURRENCY_TO_USD)
    return price * rates

# On real Etsy data, the dedicated "materials" field is often left blank --
# only ~34% of a sample pull had it filled in -- but sellers frequently
# still mention the material in tags or the title. This is a fallback
# vocabulary to recover that signal instead of defaulting straight to
# "Unknown" for the majority of listings.
MATERIAL_KEYWORDS = [
    "organic cotton", "cotton", "linen", "merino wool", "wool", "silk",
    "denim", "leather", "hemp", "recycled polyester", "polyester",
    "cashmere", "velvet", "corduroy", "bamboo", "nylon", "rayon",
    "spandex", "acrylic", "fleece", "satin", "chiffon", "lace",
    "alpaca", "viscose", "jersey", "crochet", "knit", "suede", "fur",
]


def guess_material_from_text(tags: list, title: str) -> str:
    """Scans tags + title for a known material word. Longer/more specific
    keywords are checked first (e.g. "organic cotton" before "cotton") so
    the more descriptive match wins."""
    text = (" ".join(tags) + " " + (title or "")).lower()
    for keyword in MATERIAL_KEYWORDS:
        if keyword in text:
            return keyword
    return "Unknown"


# "Dresses" had by far the worst per-category MAE of any garment bucket
# ($144.61, more than double the next-worst) -- not from bad data, but
# because it genuinely spans two very different price populations: a
# $20 casual sundress and a $4,788 atelier bridal gown are both just
# "Dresses" otherwise. Splitting on title/tags text separates these into
# two more homogeneous populations, the same fix in spirit as the
# garment-type bucketing did for near-duplicate taxonomy nodes.
BRIDAL_FORMAL_KEYWORDS = [
    "wedding", "bridal", "bridesmaid", "gown", "prom", "quinceanera",
    "quinceañera", "formal", "evening dress", "ball gown", "debutante",
]


def refine_dress_category(category: str, title: str, tags: list) -> str:
    """Splits the 'Dresses' category into Bridal/Formal vs. Casual based on
    title/tags text. Every other category passes through unchanged."""
    if category != "Dresses":
        return category
    text = f"{title or ''} {' '.join(tags)}".lower()
    if any(kw in text for kw in BRIDAL_FORMAL_KEYWORDS):
        return "Dresses (Bridal/Formal)"
    return "Dresses (Casual)"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns a raw listings dataframe (the schema produced by etsy_collector.py
    or generate_sample_data.py) into a modeling-ready feature frame.
    Keeps 'price', 'num_favorers', 'views' alongside so callers can build
    targets/weights; drop them before fitting if not needed.
    """
    out = pd.DataFrame(index=df.index)

    materials_split = df["materials"].fillna("").apply(
        lambda s: [m.strip() for m in s.split("|") if m.strip()]
    )
    tags_split = df.get("tags", pd.Series([""] * len(df))).fillna("").apply(
        lambda s: [t.strip() for t in s.split("|") if t.strip()]
    )
    titles = df.get("title", pd.Series([""] * len(df))).fillna("")

    # category = last segment of the pipe-joined category_path, with
    # Dresses further split into Bridal/Formal vs. Casual (see
    # refine_dress_category docstring for why).
    base_category = df["category_path"].fillna("Unknown").apply(
        lambda p: p.split("|")[-1].strip() or "Unknown"
    )
    out["category"] = [
        refine_dress_category(cat, title, tags)
        for cat, title, tags in zip(base_category, titles, tags_split)
    ]

    out["primary_material"] = [
        materials[0] if materials else guess_material_from_text(tags, title)
        for materials, tags, title in zip(materials_split, tags_split, titles)
    ]
    out["material_count"] = materials_split.apply(len).clip(lower=1)

    out["region"] = df["region"].fillna("Unknown").replace("", "Unknown")
    out["who_made"] = df.get("who_made", pd.Series(["i_did"] * len(df))).fillna("i_did")
    out["when_made"] = df.get("when_made", pd.Series(["made_to_order"] * len(df))).fillna("made_to_order")

    quantity = df.get("quantity", pd.Series([1] * len(df))).fillna(1).astype(float)
    out["log_quantity"] = np.log1p(quantity)

    # Carried through for target/weight construction, not used as X features directly
    if "price" in df.columns:
        if "currency_code" in df.columns:
            out["price"] = convert_to_usd(df["price"], df["currency_code"])
        else:
            out["price"] = df["price"]
    out["num_favorers"] = df.get("num_favorers", pd.Series([0] * len(df))).fillna(0)
    out["views"] = df.get("views", pd.Series([0] * len(df))).fillna(0)

    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",  # passes NUMERIC_COLS through, order preserved after cat cols
    )


def make_inference_row(category: str, materials: list, region: str,
                        when_made: str = "made_to_order",
                        quantity: int = 1) -> pd.DataFrame:
    """Builds a single-row feature frame for a brand-new (unlisted) item.
    who_made isn't a feature here -- training data is filtered to
    who_made == "i_did" only (see train_model.py), so every prediction is
    implicitly already scoped to designer-made items."""
    return pd.DataFrame([{
        "category": category,
        "primary_material": materials[0] if materials else "Unknown",
        "material_count": max(1, len(materials)),
        "region": region,
        "when_made": when_made,
        "log_quantity": np.log1p(quantity),
    }])
