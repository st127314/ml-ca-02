"""A2 page: the same form, served by the from-scratch linear regression.

Deliberately a near-clone of the A1 page so the two are comparable side by side.
The differences are the model behind it and the explanation at the top.
"""

import dash
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from model_utils import (
    OWNER_OPTIONS,
    category_options,
    dropdown_component,
    field_label,
    number_component,
)
from scratch_model_utils import config, predict_price, test_metrics, top_coefficients

dash.register_page(__name__, path="/predict-scratch", name="Predict (new model)")


def _metric_row(label, value):
    return html.Div(
        [html.Span(label, className="metric-label"), html.Span(value, className="metric-value")],
        className="metric-row",
    )


layout = html.Div(
    [
        html.Div(
            [
                html.P("CHAKY COMPANY - NEW MODEL", className="eyebrow"),
                html.H1("Used-car price predictor v2"),
                html.P(
                    "This page uses a linear regression written from scratch: the "
                    "weights are learned by gradient descent, not by a library. The "
                    "A1 page is still available in the navigation above if you want "
                    "to compare the two on the same car."
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("How this model differs from the A1 one"),
                        html.P(
                            "A1 used a Random Forest: 200 decision trees averaged "
                            "together. Scored on the same held-out cars it is the more "
                            "accurate of the two - log-scale R² 0.909 against 0.876, and "
                            "73.5% of cars priced within 20% of what they sold for against "
                            "64.8% - but it is an 18 MB file whose reasoning you cannot read."
                        ),
                        html.P(
                            "This model is a single equation. Every feature has one "
                            "coefficient saying how much it moves the price and in "
                            "which direction, so a prediction can be explained to a "
                            "customer line by line. It is also small enough to fit in "
                            "a few kilobytes."
                        ),
                        html.P(
                            "On the held-out test set, scored on the log scale the "
                            "model was actually fitted and cross-validated on:"
                        ),
                        html.Div(
                            [
                                _metric_row("R² (log scale)", f"{test_metrics['r2_log']:.3f}"),
                                _metric_row("MSE (log scale)", f"{test_metrics['mse_log']:.4f}"),
                            ],
                            className="metric-block",
                        ),
                        html.P(
                            "Being straight about the weakness: on the raw price scale "
                            "this model scores badly, and one car is responsible for "
                            "almost all of it. The single most expensive car in the test "
                            "set is over-predicted by a factor of eight, and because "
                            "price-scale error is squared after undoing the log, that one "
                            "row swamps the other 1,360. Excluding it, price-scale R² is "
                            "0.87. The Random Forest on the A1 page is steadier at the "
                            "top of the market; this model is the one that can tell you "
                            "why it picked a number.",
                            className="helper-text",
                        ),
                        html.P(
                            f"Chosen by cross-validation over 144 configurations: "
                            f"{config.get('model')} regression, "
                            f"{config.get('method')} gradient descent, "
                            f"{config.get('init')} initialisation, "
                            f"learning rate {config.get('lr')}, "
                            f"{'with' if config.get('momentum') else 'without'} momentum.",
                            className="helper-text",
                        ),
                    ],
                    className="card explain-card",
                ),
            ],
            className="content-grid single",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Vehicle details"),
                        html.P(
                            "Year and max power are required. Anything else may be left "
                            "blank and the pipeline imputes it, the same as on the A1 page."
                        ),
                        html.Div(
                            [
                                dropdown_component(
                                    "v2-brand", "Brand", category_options["brand"],
                                    "Select a brand (optional)",
                                ),
                                number_component("v2-year", "Year (required)",
                                                 placeholder="e.g. 2015", min=1980, max=2030),
                                number_component("v2-km_driven", "Kilometres driven", min=0),
                                dropdown_component("v2-fuel", "Fuel", category_options["fuel"],
                                                   "Select fuel type (optional)"),
                                dropdown_component("v2-seller_type", "Seller type",
                                                   category_options["seller_type"],
                                                   "Select seller type (optional)"),
                                dropdown_component("v2-transmission", "Transmission",
                                                   category_options["transmission"],
                                                   "Select transmission (optional)"),
                                html.Div(
                                    [
                                        field_label("Owner history"),
                                        dcc.Dropdown(
                                            id="v2-owner", options=OWNER_OPTIONS,
                                            placeholder="Select owner history (optional)",
                                            clearable=True,
                                        ),
                                    ],
                                    className="form-field",
                                ),
                                number_component("v2-mileage", "Mileage (kmpl)", min=0),
                                number_component("v2-engine", "Engine (CC)", min=0),
                                number_component("v2-max_power", "Max power (bhp, required)",
                                                 placeholder="e.g. 82", min=0),
                                number_component("v2-seats", "Seats", min=1, max=20),
                            ],
                            className="form-grid",
                        ),
                        html.Button("Predict selling price", id="v2-predict-button",
                                    n_clicks=0, className="predict-button"),
                    ],
                    className="card form-card",
                ),
                html.Div(
                    [
                        html.P("YOUR ESTIMATE", className="eyebrow"),
                        html.Div(
                            "Complete the form and click predict.",
                            id="v2-prediction-output",
                            className="prediction-output",
                        ),
                        html.P(
                            "Returned in the original selling-price scale after "
                            "reversing the model's log transformation.",
                            className="helper-text",
                        ),
                    ],
                    className="card result-card",
                ),
            ],
            className="content-grid",
        ),
    ],
    className="page",
)


@callback(
    Output("v2-prediction-output", "children"),
    Input("v2-predict-button", "n_clicks"),
    State("v2-brand", "value"),
    State("v2-year", "value"),
    State("v2-km_driven", "value"),
    State("v2-fuel", "value"),
    State("v2-seller_type", "value"),
    State("v2-transmission", "value"),
    State("v2-owner", "value"),
    State("v2-mileage", "value"),
    State("v2-engine", "value"),
    State("v2-max_power", "value"),
    State("v2-seats", "value"),
)
def predict_price_v2(n_clicks, brand, year, km_driven, fuel, seller_type,
                     transmission, owner, mileage, engine, max_power, seats):
    if not n_clicks:
        raise PreventUpdate

    # Same guard as the A1 page: these two features carry most of the signal, and
    # without them the imputer would quietly invent a median car and the page would
    # hand back a confident number built from nothing.
    missing_required = [
        label for label, value in (("year", year), ("max power", max_power)) if value is None
    ]
    if missing_required:
        return html.Div(
            f"Please enter {' and '.join(missing_required)} to get an estimate.",
            className="result-warning",
        )

    input_row = pd.DataFrame([{
        "brand": brand, "year": year, "km_driven": km_driven, "fuel": fuel,
        "seller_type": seller_type, "transmission": transmission, "owner": owner,
        "mileage": mileage, "engine": engine, "max_power": max_power, "seats": seats,
    }])

    prediction = predict_price(input_row)
    drivers = top_coefficients(5)

    return html.Div(
        [
            html.P("Estimated selling price", className="result-label"),
            html.H2(f"{prediction:,.0f}", className="price-value"),
            html.P("What this model weighs most", className="result-label"),
            html.Ul(
                [
                    html.Li(f"{name}: {'+' if coef > 0 else ''}{coef:.3f}")
                    for name, coef in drivers
                ],
                className="coef-list",
            )
            if drivers
            else html.Span(),
        ]
    )
