"""
Trains the price regression model.

Usage:
    python src/train_model.py --data data/raw/listings.csv
    python src/train_model.py                      # auto-picks raw data if present, else the sample set

Produces:
    models/price_model.joblib      -- fitted sklearn Pipeline (preprocessing + XGBoost)
    models/metrics.json            -- held-out evaluation metrics
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ALL_FEATURE_COLS, CATEGORICAL_COLS, engineer_features

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
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def evaluate(name: str, y_true_log, y_pred_log) -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    metrics = {
        "mae_usd": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 2),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }
    print(f"[{name}] MAE=${metrics['mae_usd']}  MAPE={metrics['mape_pct']}%  R2={metrics['r2']}")
    return metrics


def main(data_path: str) -> None:
    df_raw = pd.read_csv(data_path)
    df = engineer_features(df_raw)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    X = df[ALL_FEATURE_COLS]
    y_log = np.log1p(df["price"])
    # Demand-weighted training: listings the market already validated (more
    # favorites) count more. See features.py docstring for the rationale.
    sample_weight = np.log1p(df["num_favorers"].clip(lower=0)) + 1.0

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_log, sample_weight, test_size=0.2, random_state=42
    )

    results = {}

    # Baseline for comparison
    baseline = build_pipeline(Ridge(alpha=1.0))
    baseline.fit(X_train, y_train, model__sample_weight=w_train)
    results["ridge_baseline"] = evaluate("Ridge baseline", y_test, baseline.predict(X_test))

    # Core model: gradient boosting, as specified in the project brief
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
    results["xgboost"] = evaluate("XGBoost", y_test, xgb_pipeline.predict(X_test))

    # Residual spread (in log space) feeds the app's price-range estimate
    resid = y_test.values - xgb_pipeline.predict(X_test)
    results["xgboost"]["log_residual_std"] = round(float(np.std(resid)), 4)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_pipeline, MODELS_DIR / "price_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {MODELS_DIR / 'price_model.joblib'} and {MODELS_DIR / 'metrics.json'}")

    # Comparables table for the app: one row per listing, human-readable columns
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    materials_list = df_raw["materials"].fillna("").apply(
        lambda s: [m.strip() for m in s.split("|") if m.strip()]
    )
    comparables = pd.DataFrame({
        "title": df_raw.get("title", ""),
        "price": df_raw["price"],
        "category": df["category"],
        "materials": materials_list,
        "region": df["region"],
        "num_favorers": df["num_favorers"],
        "views": df["views"],
        "url": df_raw.get("url", ""),
        "image_url": df_raw.get("image_url", ""),
    }).dropna(subset=["price"])
    comparables_path = PROCESSED_DIR / "comparables.csv"
    comparables.to_csv(comparables_path, index=False)
    print(f"Saved {comparables_path} ({len(comparables)} rows)")

    # Quick feature-importance readout
    ohe = xgb_pipeline.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    numeric_names = [c for c in ALL_FEATURE_COLS if c not in CATEGORICAL_COLS]
    feature_names = cat_feature_names + numeric_names
    importances = xgb_pipeline.named_steps["model"].feature_importances_
    top = sorted(zip(feature_names, importances), key=lambda t: -t[1])[:10]
    print("\nTop features:")
    for name, imp in top:
        print(f"  {name:35s} {imp:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to a listings CSV")
    args = parser.parse_args()
    main(args.data or pick_default_data_path())
