from pathlib import Path

import joblib
from dash import dcc, html

APP_DIR = Path(__file__).resolve().parent
# stored zlib-compressed: a plain pickle of this forest is 152 MB, over GitHub's
# 100 MB file limit, while the compressed dump is ~35 MB and identical once loaded
MODEL_PATH = APP_DIR.parent / "models" / "car_price_model.joblib"

model = joblib.load(MODEL_PATH)

# pull the categories the OneHotEncoder learned so the dropdowns match training data
preprocessor = model.named_steps["preprocessor"]
categorical_transformer = next(
    transformer
    for name, transformer, _ in preprocessor.transformers_
    if name == "categorical"
)
categorical_columns = next(
    columns
    for name, _, columns in preprocessor.transformers
    if name == "categorical"
)
category_values = categorical_transformer.named_steps["onehot"].categories_
category_options = {
    column: sorted(values.tolist())
    for column, values in zip(categorical_columns, category_values)
}

OWNER_OPTIONS = [
    {"label": "First Owner", "value": 1},
    {"label": "Second Owner", "value": 2},
    {"label": "Third Owner", "value": 3},
    {"label": "Fourth & Above Owner", "value": 4},
]


def field_label(text):
    return html.Label(text, className="field-label")


def dropdown_component(component_id, label, options, placeholder):
    return html.Div(
        [
            field_label(label),
            dcc.Dropdown(
                id=component_id,
                options=[{"label": option, "value": option} for option in options],
                placeholder=placeholder,
                clearable=True,
            ),
        ],
        className="form-field",
    )


def number_component(component_id, label, placeholder="Optional", **kwargs):
    return html.Div(
        [
            field_label(label),
            dcc.Input(
                id=component_id,
                type="number",
                placeholder=placeholder,
                **kwargs,
            ),
        ],
        className="form-field",
    )
