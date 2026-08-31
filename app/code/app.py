import os

import dash
from dash import Dash, dcc, html

app = Dash(__name__, use_pages=True, title="Chaky Car Price Predictor")
server = app.server

app.layout = html.Div(
    [
        html.Nav(
            [
                html.Span("CHAKY", className="nav-brand"),
                html.Div(
                    [
                        dcc.Link(page["name"], href=page["path"], className="nav-link")
                        for page in dash.page_registry.values()
                    ],
                    className="nav-links",
                ),
            ],
            className="navbar",
        ),
        dash.page_container,
    ]
)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(debug=False, host=host, port=8050, use_reloader=False)
