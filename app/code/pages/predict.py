import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from model_utils import (
    OWNER_OPTIONS,
    category_options,
    dropdown_component,
    field_label,
    model,
    number_component,
)

dash.register_page(__name__, path="/predict", name="Predict (A1 model)")


layout = html.Div(
    [
        html.Div(
            [
                html.P("CHAKY COMPANY", className="eyebrow"),
                html.H1("Used-car price predictor"),
                html.P(
                    "Year and max power are required - they carry most of the "
                    "prediction. Leave any other field blank and the model will "
                    "impute a sensible value from the training data."
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Vehicle details"),
                        html.P(
                            "CNG and LPG cars are not supported because their mileage "
                            "uses a different measurement system."
                        ),
                        html.Div(
                            [
                                dropdown_component(
                                    "brand",
                                    "Brand",
                                    category_options["brand"],
                                    "Select a brand (optional)",
                                ),
                                number_component("year", "Year (required)", placeholder="e.g. 2015", min=1980, max=2030),
                                number_component("km_driven", "Kilometres driven", min=0),
                                dropdown_component(
                                    "fuel",
                                    "Fuel",
                                    category_options["fuel"],
                                    "Select fuel type (optional)",
                                ),
                                dropdown_component(
                                    "seller_type",
                                    "Seller type",
                                    category_options["seller_type"],
                                    "Select seller type (optional)",
                                ),
                                dropdown_component(
                                    "transmission",
                                    "Transmission",
                                    category_options["transmission"],
                                    "Select transmission (optional)",
                                ),
                                html.Div(
                                    [
                                        field_label("Owner history"),
                                        dcc.Dropdown(
                                            id="owner",
                                            options=OWNER_OPTIONS,
                                            placeholder="Select owner history (optional)",
                                            clearable=True,
                                        ),
                                    ],
                                    className="form-field",
                                ),
                                number_component("mileage", "Mileage (kmpl)", min=0),
                                number_component("engine", "Engine (CC)", min=0),
                                number_component("max_power", "Max power (bhp, required)", placeholder="e.g. 82", min=0),
                                number_component("seats", "Seats", min=1, max=20),
                            ],
                            className="form-grid",
                        ),
                        html.Button(
                            "Predict selling price",
                            id="predict-button",
                            n_clicks=0,
                            className="predict-button",
                        ),
                    ],
                    className="card form-card",
                ),
                html.Div(
                    [
                        html.P("YOUR ESTIMATE", className="eyebrow"),
                        html.Div(
                            "Complete the form and click predict.",
                            id="prediction-output",
                            className="prediction-output",
                        ),
                        html.P(
                            "The prediction is returned in the original selling-price "
                            "scale after reversing the model's log transformation.",
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
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("brand", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
)
def predict_price(
    n_clicks,
    brand,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    mileage,
    engine,
    max_power,
    seats,
):
    if not n_clicks:
        raise PreventUpdate

    # the two features the model leans on hardest; without them the pipeline would just
    # impute the median car and hand back a confident-looking price built from nothing
    missing_required = [
        label
        for label, value in (("year", year), ("max power", max_power))
        if value is None
    ]
    if missing_required:
        return html.Div(
            f"Please enter {' and '.join(missing_required)} to get an estimate.",
            className="result-warning",
        )

    input_row = pd.DataFrame(
        [
            {
                "brand": brand,
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "seats": seats,
            }
        ]
    )

    # blank fields become None here, pipeline's imputer fills them in
    prediction_log = float(model.predict(input_row)[0])
    prediction = float(np.exp(prediction_log))

    return html.Div(
        [
            html.P("Estimated selling price", className="result-label"),
            html.H2(f"{prediction:,.0f}", className="price-value"),
        ]
    )
