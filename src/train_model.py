"""
Trains the price regression model.

Usage:
    python src/train_model.py --data data/raw/listings.csv
    python src/train_model.py                      # auto-picks raw data if present, else the sample set

Produces:
    models/price_model.joblib      -- fitted sklearn Pipeline (preprocessing + XGBoost)
    models/metrics.json            -- held-out evaluation metrics, baseline comparison,
                                       per-category breakdown, and interval coverage
    data/processed/comparables.csv -- cleaned listings table the Streamlit app searches for comparables
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ALL_FEATURE_COLS, CATEGORICAL_COLS, engineer_features, guess_material_from_text

# Anchor all paths to the project root (parent of src/) so this script gives
# identical results whether it's run as `python src/train_model.py` from the
# repo root or `python train_model.py` from inside src/.
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def pick_default_data_path() -> str:
    raw = ROOT_DIR / "data" / "raw" / "listings.csv"
    sample = ROOT_DIR / "data" / "sample" / "listings_sample.csv"
    if raw.exists():
        print(f"Using real collected data: {raw}")
        return str(raw)
    print(f"No {raw} found -- falling back to synthetic sample data: {sample}")
    return str(sample)


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[(
            "cat",
            # min_frequency=25: any category (region, material, etc.)
            # appearing fewer than 25 times gets pooled into a single
            # "infrequent" bucket instead of becoming its own one-hot
            # column. Found this was needed after region ballooned to 55
            # distinct values with a long thin tail (many countries with
            # under 10 listings each) -- same overfitting shape as the
            # "knitted" material issue from earlier, just in a different
            # column. handle_unknown="infrequent_if_exist" routes any
            # brand-new category at inference time into the same bucket
            # rather than an all-zero encoding.
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=25),
            CATEGORICAL_COLS,
        )],
        remainder="passthrough",
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def evaluate(name: str, y_true_log, y_pred_log) -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    metrics = {
        "mae_usd": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse_usd": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "median_ae_usd": round(float(median_absolute_error(y_true, y_pred)), 2),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 2),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }
    print(f"[{name}] MAE=${metrics['mae_usd']}  RMSE=${metrics['rmse_usd']}  "
          f"MedianAE=${metrics['median_ae_usd']}  MAPE={metrics['mape_pct']}%  R2={metrics['r2']}")
    return metrics


def per_category_breakdown(category: pd.Series, y_true_log, y_pred_log) -> dict:
    """MAE broken out by garment category, to see where the model is
    actually solid vs. where it's shaky -- a single blended MAE hides
    this entirely."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.asarray(y_pred_log))
    df = pd.DataFrame({"category": category.values, "y_true": y_true.values, "y_pred": y_pred})
    breakdown = {}
    print("\nPer-category MAE (test set):")
    for cat, group in df.groupby("category"):
        mae = float(mean_absolute_error(group["y_true"], group["y_pred"]))
        breakdown[cat] = {"mae_usd": round(mae, 2), "n_test_rows": len(group)}
        print(f"  {cat:25s} MAE=${mae:7.2f}  (n={len(group)})")
    return breakdown


def prediction_interval_coverage(resid_log: np.ndarray, residual_std: float) -> float:
    """What fraction of test-set actual prices fall inside the +/-1 std
    range the app shows as its 'confidence range'? For a roughly normal
    residual distribution this should land near ~68%.

    Honesty check on the method: residual_std is computed FROM this same
    test set's residuals, so this isn't a fully independent generalization
    check (a proper 3-way train/calibration/test split would be more
    rigorous). What it DOES legitimately test is whether the residuals are
    roughly normally shaped -- which is the assumption the app's symmetric
    +/-1-std range relies on. If coverage lands far from ~68%, that's a
    real signal the range should be built from empirical quantiles instead
    of a std-based formula, not that anything is "broken."
    """
    within = np.abs(resid_log) <= residual_std
    return float(np.mean(within))


def main(data_path: str) -> None:
    df_raw = pd.read_csv(data_path)
    df = engineer_features(df_raw)

    # Scope to items the seller actually made themselves. The project is
    # specifically about independent/student designers pricing their own
    # handmade work -- resale, vintage, and collective/supply listings
    # (who_made != "i_did") are a different market and would bias the
    # model toward prices that aren't relevant comparables (a $2,090
    # resale "CUSTOM MADE Oscar de la Renta" piece isn't a fair comp for
    # someone pricing a piece they made themselves).
    before_who_made = len(df)
    df = df[df["who_made"] == "i_did"]
    if before_who_made != len(df):
        print(f"Dropped {before_who_made - len(df)} rows where who_made != 'i_did' "
              f"(resale/vintage/collective listings, not designer-made items)")

    before = len(df)
    df_raw_aligned_pre = df_raw.loc[df.index]
    dropped_currency = df["price"].isna() & df_raw_aligned_pre["price"].notna()
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    if before != len(df):
        print(f"Dropped {before - len(df)} rows (missing price, non-positive price, "
              f"or an unmapped currency -- {dropped_currency.sum()} were unmapped currencies)")

    X = df[ALL_FEATURE_COLS]
    y_log = np.log1p(df["price"])
    # Demand-weighted training: listings the market already validated (more
    # favorites) count more. See features.py docstring for the rationale.
    sample_weight = np.log1p(df["num_favorers"].clip(lower=0)) + 1.0

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_log, sample_weight, test_size=0.2, random_state=42
    )

    results = {}

    # Naive baseline: predict the overall training-set median price for
    # every test row, ignoring every feature. This answers "is the model
    # actually better than knowing nothing about the item at all?" -- the
    # bar every other model needs to clear to justify its complexity.
    naive_median_log = float(np.median(y_train))
    naive_pred_log = np.full(len(y_test), naive_median_log)
    results["naive_median_baseline"] = evaluate("Naive median baseline", y_test, naive_pred_log)

    # Linear baseline
    ridge = build_pipeline(Ridge(alpha=1.0))
    ridge.fit(X_train, y_train, model__sample_weight=w_train)
    results["ridge_baseline"] = evaluate("Ridge baseline", y_test, ridge.predict(X_test))

    # Random Forest: a second tree-based baseline, so XGBoost's result can
    # be judged against "another tree ensemble" and not just a linear model.
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_pipeline = build_pipeline(rf_model)
    rf_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    rf_pred_log = rf_pipeline.predict(X_test)
    results["random_forest"] = evaluate("Random Forest", y_test, rf_pred_log)

    # Gradient boosting, as specified in the project brief
    xgb_model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    xgb_pipeline = build_pipeline(xgb_model)
    xgb_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    xgb_pred_log = xgb_pipeline.predict(X_test)
    results["xgboost"] = evaluate("XGBoost", y_test, xgb_pred_log)

    # Deploy whichever tree-based model actually performs best on held-out
    # data, rather than always defaulting to XGBoost regardless of result.
    # Restricted to Random Forest vs. XGBoost (not Ridge): SHAP's
    # TreeExplainer, which powers this project's explainability feature,
    # only supports tree-based models -- Ridge stays in the comparison
    # table as context but was never a deployable candidate.
    candidates = {
        "Random Forest": (rf_pipeline, rf_pred_log, "random_forest"),
        "XGBoost": (xgb_pipeline, xgb_pred_log, "xgboost"),
    }
    best_name, (best_pipeline, best_pred_log, best_key) = max(
        candidates.items(), key=lambda item: results[item[1][2]]["r2"]
    )
    other_name = "XGBoost" if best_name == "Random Forest" else "Random Forest"
    other_r2 = results[candidates[other_name][2]]["r2"]
    print(f"\nDeploying: {best_name} (R2={results[best_key]['r2']}, vs. {other_name}={other_r2})")
    results["deployed_model"] = best_name

    # Residual spread (in log space) feeds the app's price-range estimate
    resid = y_test.values - best_pred_log
    residual_std = float(np.std(resid))
    results[best_key]["log_residual_std"] = round(residual_std, 4)

    # Prediction interval coverage -- see function docstring for exactly
    # what this does and doesn't prove.
    coverage = prediction_interval_coverage(resid, residual_std)
    results[best_key]["interval_coverage_pct"] = round(coverage * 100, 1)
    print(f"\nPrediction interval coverage: {coverage*100:.1f}% of test-set actual prices "
          f"fell within the +/-1-std confidence range (target: ~68% if residuals are "
          f"roughly normal)")

    # Per-category error breakdown -- where is the model actually solid
    # vs. still data-starved?
    results[best_key]["per_category_mae"] = per_category_breakdown(
        X_test["category"], y_test, best_pred_log
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "price_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {MODELS_DIR / 'price_model.joblib'} and {MODELS_DIR / 'metrics.json'}")

    # Comparables table for the app: one row per listing, human-readable columns
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Align raw fields to the same rows that survived filtering in df (price
    # cleaning + currency conversion may have dropped some) -- mixing the
    # full df_raw with the filtered df here previously produced silent NaN
    # categories for any row that got dropped.
    df_raw_aligned = df_raw.loc[df.index]
    materials_list = df_raw_aligned["materials"].fillna("").apply(
        lambda s: [m.strip() for m in s.split("|") if m.strip()]
    )
    tags_list = df_raw_aligned.get("tags", pd.Series([""] * len(df_raw_aligned))).fillna("").apply(
        lambda s: [t.strip() for t in s.split("|") if t.strip()]
    )
    titles = df_raw_aligned.get("title", pd.Series([""] * len(df_raw_aligned))).fillna("")
    materials_list = [
        [guess_material_from_text(m + t, title)]
        for m, t, title in zip(materials_list, tags_list, titles)
    ]
    comparables = pd.DataFrame({
        "title": df_raw_aligned.get("title", ""),
        "price": df["price"],  # already converted to USD
        "category": df["category"],
        "materials": materials_list,
        "region": df["region"],
        "num_favorers": df["num_favorers"],
        "views": df["views"],
        "url": df_raw_aligned.get("url", ""),
        "image_url": df_raw_aligned.get("image_url", ""),
    }).dropna(subset=["price"])
    comparables_path = PROCESSED_DIR / "comparables.csv"
    comparables.to_csv(comparables_path, index=False)
    print(f"Saved {comparables_path} ({len(comparables)} rows)")

    # Quick feature-importance readout -- from whichever model was actually
    # deployed, not always XGBoost. Both RandomForestRegressor and
    # XGBRegressor expose .feature_importances_, so this works either way.
    ohe = best_pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    numeric_names = [c for c in ALL_FEATURE_COLS if c not in CATEGORICAL_COLS]
    feature_names = cat_feature_names + numeric_names
    importances = best_pipeline.named_steps["model"].feature_importances_
    top = sorted(zip(feature_names, importances), key=lambda t: -t[1])[:10]
    print(f"\nTop features ({best_name}):")
    for name, imp in top:
        print(f"  {name:35s} {imp:.3f}")

    # Baseline comparison summary -- the table that answers "did the
    # deployed model actually earn its complexity?"
    print("\nModel comparison (R2, higher is better):")
    print(f"  {'Naive median baseline':25s} {results['naive_median_baseline']['r2']:>7.3f}")
    print(f"  {'Ridge':25s} {results['ridge_baseline']['r2']:>7.3f}")
    rf_marker = "  <- deployed" if best_name == "Random Forest" else ""
    xgb_marker = "  <- deployed" if best_name == "XGBoost" else ""
    print(f"  {'Random Forest':25s} {results['random_forest']['r2']:>7.3f}{rf_marker}")
    print(f"  {'XGBoost':25s} {results['xgboost']['r2']:>7.3f}{xgb_marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to a listings CSV")
    args = parser.parse_args()
    main(args.data or pick_default_data_path())
