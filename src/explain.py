"""
SHAP-based explainability for the price model.

Turns "the model predicts $84" into "the model predicts $84 because:
Material +$12, Category +$8, Region +$3, ..." -- decision support instead
of an opaque number, per the reframing from "fair price" to "typical
range with visible reasoning."

A NOTE ON THE DOLLAR CONVERSION
--------------------------------
The model predicts log1p(price), so raw SHAP values come out in log-space
-- additive contributions to the *log* prediction, not dollars directly.
Naively multiplying each log-space SHAP value by the final price (a
tempting shortcut) does NOT work: it produces contributions that don't
sum to the actual price, sometimes wildly so, because it ignores that
exp() is nonlinear. (Caught this in testing: for one example it produced
contributions summing to $198 against an actual prediction of $81.)

Instead, this applies each feature's log-space contribution one at a time
to a running total and measures the *actual* dollar change at each step
(a "waterfall" decomposition). This guarantees the displayed
contributions sum exactly to (final price - baseline price), which is
what makes the breakdown trustworthy to show next to the final number.
The trade-off: because exp() is nonlinear, the dollar size of each
feature's contribution technically depends on what order it's applied
in relative to the others -- a known, unavoidable property of converting
an additive log-space explanation into dollar terms, not a bug. Order
here is fixed (dict insertion order) so results are at least consistent
across repeated calls for the same input.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from features import CATEGORICAL_COLS, NUMERIC_COLS

# Maps internal feature-group names to human-readable labels for display.
# One-hot columns (e.g. "category_Dresses", "category_Coats", ...) all
# collapse back to one label ("Category") since a single prediction only
# ever has one of them active -- showing "Category: +$8" is more readable
# than "category_Dresses: +$8".
FEATURE_GROUP_LABELS = {
    "category": "Category",
    "primary_material": "Material",
    "region": "Region",
    "when_made": "Listing age/type",
    "material_count": "Material count",
    "log_quantity": "Quantity",
}


def explain_prediction(pipeline, row: pd.DataFrame) -> tuple:
    """Returns ({human_label: dollar_contribution}, baseline_price) for one
    input row. baseline_price + sum(contributions.values()) == the model's
    actual predicted price, by construction."""
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(row)
    # XGBoost treats a sparse matrix's implicit zeros as *missing values*,
    # not literal 0.0 -- densifying silently changes what it predicts
    # (confirmed by testing: produced a $198 vs. $81 mismatch before this
    # was caught). So XGBoost must stay sparse.
    #
    # scikit-learn's RandomForestRegressor has no such distinction --
    # sparse and dense give it byte-identical predictions (confirmed by
    # testing) -- but SHAP's TreeExplainer path for RandomForest crashes
    # on sparse input outright. So RF needs densifying and XGBoost must
    # NOT be densified; this project now deploys whichever model wins on
    # a given dataset, so both paths have to work correctly, not just one.
    if isinstance(model, RandomForestRegressor):
        transformed = transformed.toarray()

    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    feature_names = cat_feature_names + NUMERIC_COLS

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)[0]
    # expected_value comes back as a plain scalar for XGBoost but as an
    # array for RandomForestRegressor -- np.ravel(...)[0] handles both.
    base_log = float(np.ravel(explainer.expected_value)[0])

    # Group SHAP values (still in log-space) by original feature, before
    # converting to dollars
    grouped_log_contributions = {}
    for name, value in zip(feature_names, shap_values):
        group = next((c for c in CATEGORICAL_COLS if name.startswith(c + "_")), name)
        label = FEATURE_GROUP_LABELS.get(group, group)
        grouped_log_contributions[label] = grouped_log_contributions.get(label, 0.0) + float(value)

    # Telescoping conversion to dollars -- see module docstring for why
    # this replaced a naive linear scaling that didn't sum correctly.
    running_log = base_log
    baseline_price = float(np.expm1(running_log))
    running_price = baseline_price
    dollar_contributions = {}
    for label, log_value in grouped_log_contributions.items():
        running_log += log_value
        new_price = float(np.expm1(running_log))
        dollar_contributions[label] = new_price - running_price
        running_price = new_price

    return dollar_contributions, baseline_price
