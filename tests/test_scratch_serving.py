"""The serving path for the A2 model: artifact loads, and blanks are handled."""

import numpy as np
import pandas as pd
import pytest

from scratch_model_utils import config, predict_price, test_metrics, top_coefficients

BASE = dict(
    year=2015, km_driven=50000, owner=1,
    mileage=20.0, engine=1200.0, max_power=82.0, seats=5.0,
)
CATEGORICAL = ["brand", "fuel", "seller_type", "transmission"]


def row(**overrides):
    values = {c: None for c in CATEGORICAL}
    values.update(BASE)
    values.update(overrides)
    return pd.DataFrame([values])


def test_artifact_carries_its_metadata():
    assert {"mse_log", "r2_log"} <= set(test_metrics)
    assert {"model", "method", "init", "lr"} <= set(config)


def test_fully_specified_car_gets_a_plausible_price():
    price = predict_price(row(brand="Maruti", fuel="Petrol",
                              seller_type="Individual", transmission="Manual"))
    assert 50_000 < price < 5_000_000


def test_blank_categoricals_are_imputed_not_dropped():
    """The regression this guards: None is not np.nan, so SimpleImputer leaves it
    alone and OneHotEncoder emits an all-zero row. For a linear model that deletes
    the categorical coefficients from the sum and priced a real car at 246."""
    price = predict_price(row())
    assert 50_000 < price < 5_000_000


def test_blank_and_specified_agree_within_an_order_of_magnitude():
    blank = predict_price(row())
    full = predict_price(row(brand="Maruti", fuel="Petrol",
                             seller_type="Individual", transmission="Manual"))
    assert 0.2 < blank / full < 5


@pytest.mark.parametrize("year,newer", [(2005, 2019)])
def test_a_newer_car_is_worth_more(year, newer):
    assert predict_price(row(year=newer)) > predict_price(row(year=year))


def test_more_power_is_worth_more():
    assert predict_price(row(max_power=150.0)) > predict_price(row(max_power=60.0))


def test_prediction_is_finite_for_sparse_input():
    sparse = row(km_driven=None, mileage=None, engine=None, seats=None, owner=None)
    assert np.isfinite(predict_price(sparse))


def test_top_coefficients_are_sorted_by_magnitude():
    coefficients = top_coefficients(5)
    if coefficients:  # empty when polynomial expansion changes the feature count
        magnitudes = [abs(c) for _, c in coefficients]
        assert magnitudes == sorted(magnitudes, reverse=True)
