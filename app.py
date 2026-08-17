"""
Designer Pricing Assistant -- Streamlit app.

Run:
    streamlit run app.py

Enter an item's category, materials, and region; get a suggested price
range from the trained XGBoost model, backed by real comparable listings
pulled from the training data.
"""

import ast
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from features import make_inference_row

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "price_model.joblib"
COMPARABLES_PATH = ROOT_DIR / "data" / "processed" / "comparables.csv"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

st.set_page_config(page_title="Designer Pricing Assistant", page_icon="🧵", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparables() -> pd.DataFrame:
    df = pd.read_csv(COMPARABLES_PATH)
    # materials was saved as a Python-list repr string; parse it back safely
    df["materials"] = df["materials"].apply(ast.literal_eval)
    return df


@st.cache_data
def load_residual_std() -> float:
    import json
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            m = json.load(f)
        return m.get("xgboost", {}).get("log_residual_std", 0.25)
    return 0.25


def find_comparables(df: pd.DataFrame, category: str, materials: list, region: str, top_n: int = 8) -> pd.DataFrame:
    """Ranks listings by category match, region match, and material overlap."""
    selected = set(m.lower() for m in materials)

    def score(row) -> float:
        s = 0.0
        if row["category"] == category:
            s += 2.0
        if row["region"] == region:
            s += 1.0
        row_materials = set(m.lower() for m in row["materials"])
        overlap = len(selected & row_materials)
        s += overlap * 0.75
        return s

    scored = df.copy()
    scored["match_score"] = scored.apply(score, axis=1)
    scored = scored[scored["match_score"] > 0].sort_values(
        ["match_score", "num_favorers"], ascending=[False, False]
    )
    return scored.head(top_n)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧵 Designer Pricing Assistant")
st.caption(
    "For independent & student designers: enter your item's attributes and get a "
    "fair price suggestion, backed by real comparable listings."
)

if not MODEL_PATH.exists():
    st.error(
        f"No trained model found at `{MODEL_PATH}`. Run `python src/train_model.py` first "
        "(it will use the synthetic sample data unless you've collected real Etsy data)."
    )
    st.stop()

model = load_model()
comparables = load_comparables()
residual_std = load_residual_std()

categories = sorted(comparables["category"].dropna().unique())
all_materials = sorted({m for row in comparables["materials"] for m in row})
regions = sorted(comparables["region"].dropna().unique())

with st.sidebar:
    st.header("Item attributes")
    category = st.selectbox("Category", categories)
    materials = st.multiselect(
        "Materials", all_materials,
        default=[all_materials[0]] if all_materials else [],
        help="Pick the primary material first -- it carries the most weight.",
    )
    region = st.selectbox("Your region", regions)
    quantity = st.number_input("Quantity available", min_value=1, value=1)
    submitted = st.button("Get price suggestion", type="primary", width="stretch")

if not submitted:
    st.info("Set your item's attributes in the sidebar, then click **Get price suggestion**.")
    st.stop()

if not materials:
    st.warning("Pick at least one material for an accurate estimate.")
    st.stop()

# --- Model prediction -------------------------------------------------------
row = make_inference_row(category=category, materials=materials, region=region, quantity=quantity)
pred_log = model.predict(row)[0]
point_estimate = float(np.expm1(pred_log))

# Model-uncertainty range from held-out residual spread (log space)
model_low = float(np.expm1(pred_log - residual_std))
model_high = float(np.expm1(pred_log + residual_std))

# --- Real comparables --------------------------------------------------------
comps = find_comparables(comparables, category, materials, region)

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Suggested price")
    st.metric("Point estimate", f"${point_estimate:,.2f}")
    st.write(f"Model confidence range: **${model_low:,.2f} – ${model_high:,.2f}**")

    if len(comps) >= 3:
        comp_low, comp_high = np.percentile(comps["price"], [25, 75])
        st.write(f"Comparable-listings range (25th–75th pct of {len(comps)} similar items): "
                  f"**${comp_low:,.2f} – ${comp_high:,.2f}**")
    else:
        st.write("Not enough close comparables to compute a market range -- relying on the model estimate.")

    st.caption(
        "The model is trained with more weight on listings that already have engagement "
        "(favorites/views), so the price reflects patterns from items the market has "
        "actually responded to -- not just what's listed."
    )

with col2:
    st.subheader(f"Real comparable listings ({len(comps)})")
    if comps.empty:
        st.write("No close matches found for this combination yet.")
    else:
        for _, c in comps.iterrows():
            with st.container(border=True):
                cc1, cc2 = st.columns([1, 3])
                with cc1:
                    if isinstance(c.get("image_url"), str) and c["image_url"]:
                        st.image(c["image_url"], width="stretch")
                with cc2:
                    st.markdown(f"**{c['title']}** — ${c['price']:,.2f}")
                    st.caption(
                        f"{c['category']} · {', '.join(c['materials'])} · {c['region']} · "
                        f"❤ {int(c['num_favorers'])} favorites"
                    )
                    if isinstance(c.get("url"), str) and c["url"]:
                        st.markdown(f"[View listing]({c['url']})")

st.divider()
st.caption(
    "Data: Etsy Open API v3 handmade-clothing listings (or synthetic sample data if no real "
    "data has been collected yet — check the sidebar/README to swap in your own pull)."
)
