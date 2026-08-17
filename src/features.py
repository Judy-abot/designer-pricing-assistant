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

CATEGORICAL_COLS = ["category", "primary_material", "region", "who_made", "when_made"]
NUMERIC_COLS = ["material_count", "log_quantity"]

ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns a raw listings dataframe (the schema produced by etsy_collector.py
    or generate_sample_data.py) into a modeling-ready feature frame.
    Keeps 'price', 'num_favorers', 'views' alongside so callers can build
    targets/weights; drop them before fitting if not needed.
    """
    out = pd.DataFrame(index=df.index)

    # category = last segment of the pipe-joined category_path
    out["category"] = (
        df["category_path"].fillna("Unknown").apply(lambda p: p.split("|")[-1].strip() or "Unknown")
    )

    materials_split = df["materials"].fillna("").apply(
        lambda s: [m.strip() for m in s.split("|") if m.strip()]
    )
    out["primary_material"] = materials_split.apply(lambda m: m[0] if m else "Unknown")
    out["material_count"] = materials_split.apply(len).clip(lower=1)

    out["region"] = df["region"].fillna("Unknown").replace("", "Unknown")
    out["who_made"] = df.get("who_made", pd.Series(["i_did"] * len(df))).fillna("i_did")
    out["when_made"] = df.get("when_made", pd.Series(["made_to_order"] * len(df))).fillna("made_to_order")

    quantity = df.get("quantity", pd.Series([1] * len(df))).fillna(1).astype(float)
    out["log_quantity"] = np.log1p(quantity)

    # Carried through for target/weight construction, not used as X features directly
    if "price" in df.columns:
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
                        who_made: str = "i_did", when_made: str = "made_to_order",
                        quantity: int = 1) -> pd.DataFrame:
    """Builds a single-row feature frame for a brand-new (unlisted) item."""
    return pd.DataFrame([{
        "category": category,
        "primary_material": materials[0] if materials else "Unknown",
        "material_count": max(1, len(materials)),
        "region": region,
        "who_made": who_made,
        "when_made": when_made,
        "log_quantity": np.log1p(quantity),
    }])
