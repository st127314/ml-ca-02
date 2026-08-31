"""Loader for the A2 from-scratch model.

The notebook saves a dict rather than a bare estimator, because serving needs the
fitted preprocessor as well as the fitted weights. Keeping them in one file is what
stops the app from preprocessing differently than training did.

Importing `linear_regression` here is not decorative: joblib records the class's
import path, so the module has to be importable for the artifact to load at all.
"""

from pathlib import Path

import joblib
import numpy as np

import linear_regression  # noqa: F401  - required for joblib to resolve the class

APP_DIR = Path(__file__).resolve().parent
SCRATCH_MODEL_PATH = APP_DIR.parent / "models" / "car_price_scratch_model.joblib"

_artifact = joblib.load(SCRATCH_MODEL_PATH)

preprocessor = _artifact["preprocessor"]
model = _artifact["model"]
feature_names = _artifact["feature_names"]
config = _artifact["config"]
test_metrics = _artifact["test_metrics"]


def _normalise_missing(input_row):
    """Turn None into np.nan before the pipeline sees it.

    SimpleImputer decides what is missing with `x != x`, which is True for np.nan
    and False for None. A None left in an object column therefore survives
    imputation as a literal category, and OneHotEncoder(handle_unknown="ignore")
    then encodes it as an all-zero row - silently dropping that feature instead of
    filling it in.

    A tree model shrugs this off. A linear model does not: with brand, fuel,
    seller_type and transmission all zeroed, their coefficients (worth about +2.8,
    +2.6 and +1.9 in log space) simply vanish from the sum, and a car worth 424,189
    is priced at 246.
    """
    row = input_row.copy()
    for column in row.columns:
        row[column] = row[column].where(row[column].notna(), np.nan)
    return row


def predict_price(input_row):
    """Raw one-row DataFrame in, price in rupees out.

    The pipeline mirrors training exactly: transform, prepend the bias column
    (this implementation has no separate intercept - theta[0] is it), predict in
    log space, then undo the log.
    """
    matrix = np.asarray(preprocessor.transform(_normalise_missing(input_row)), dtype=float)
    matrix = np.c_[np.ones(matrix.shape[0]), matrix]
    prediction_log = float(model.predict(matrix)[0])
    return float(np.exp(prediction_log))


def top_coefficients(n=8):
    """The n features with the largest coefficients, for display on the page.

    Only interpretable because the features were standardised before fitting -
    otherwise a coefficient would reflect its feature's unit, not its importance.
    """
    coefficients = np.asarray(model._coef())
    names = np.asarray(feature_names)
    if len(names) != len(coefficients):  # polynomial expansion changes the count
        return []
    order = np.argsort(np.abs(coefficients))[::-1][:n]
    return [(str(names[i]).split("__")[-1], float(coefficients[i])) for i in order]
