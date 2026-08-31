import dash
from dash import dcc, html

dash.register_page(__name__, path="/", name="Home")


def stat_card(value, label):
    return html.Div(
        [
            html.P(value, className="stat-value"),
            html.P(label, className="stat-label"),
        ],
        className="stat-card",
    )


layout = html.Div(
    [
        html.Div(
            [
                html.P("CHAKY COMPANY", className="eyebrow"),
                html.H1("How much is your used car actually worth?"),
                html.P(
                    "Chaky sells a lot of used cars, and pricing them by gut feeling "
                    "was getting inconsistent — similar cars kept landing on wildly "
                    "different price tags. This tool is built to give the sales team "
                    "a quick, data-backed starting point before they negotiate.",
                    className="lede",
                ),
                dcc.Link(
                    html.Button("Try the predictor", className="predict-button"),
                    href="/predict",
                ),
            ],
            className="hero",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("The approach"),
                        html.P(
                            "Chaky company has over 8,000 listings scraped from a used-car "
                            "marketplace — brand, year, mileage, fuel type, engine size, "
                            "power, number of previous owners, and the price it sold for. "
                            "We first ran exploratory data analysis to understand the data, "
                            "then cleaned it up: dropped the torque column (not something "
                            "the sales team reads anyway), pulled the numbers out of fields "
                            "like \"23.4 kmpl\" and \"1248 CC\", kept only the brand name "
                            "from the full listing title, dropped CNG/LPG cars because they "
                            "report mileage in a different unit, and removed \"Test Drive "
                            "Car\" listings since those are priced completely differently "
                            "from a normal resale. Around 1,200 exactly duplicated listings "
                            "were removed too, so the model is not scored on cars it has "
                            "already memorised, which leaves about 6,800 to learn from. "
                            "Several models were then trained and compared, and the best "
                            "one was deployed as this web app. "
                            "Missing values are not dropped: the app fills them in with "
                            "values learned from the training data, which is what lets you "
                            "leave a field blank in the form."
                        ),
                        html.P(
                            "Selling price is heavily right-skewed — a handful of luxury "
                            "cars sell for many times the typical price — so the model is "
                            "trained on the log of the price instead of the raw number. "
                            "That keeps a few expensive outliers from dragging the whole "
                            "model around. Predictions are converted back to the original "
                            "price scale before they're shown to you."
                        ),
                    ],
                    className="card",
                ),
                html.Div(
                    [
                        html.H2("Picking a model"),
                        html.P(
                            "Three models were compared: a baseline that just guesses the "
                            "median price, a Ridge (linear) regression, and a Random Forest. "
                            "Random Forest came out on top on a held-out validation split, "
                            "and 5-fold cross-validation confirmed the same ranking rather "
                            "than one lucky split. A 24-combination grid search then tuned "
                            "it. That tuned model is the one running behind this app:"
                        ),
                        html.Div(
                            [
                                stat_card("80.5k", "Typical error (MAE) on unseen cars"),
                                stat_card("73%", "Cars priced within 20% of actual"),
                                stat_card("2 features", "Drive most of the prediction"),
                            ],
                            className="stat-grid",
                        ),
                        html.P(
                            "Random Forest wins here because car pricing isn't a straight "
                            "line — how much power matters, for example, depends a lot on "
                            "how old the car already is, and a linear model like Ridge "
                            "can't represent that kind of interaction on its own. Ridge "
                            "still beat the median baseline by a wide margin though, so a "
                            "good chunk of the price is explainable linearly too."
                        ),
                        html.P(
                            "Looking at what the chosen model actually leans on: year and "
                            "max power carry most of the prediction, followed by engine "
                            "size and fuel type. Brand comes next — it looks unimportant if "
                            "you count each brand separately, but judged as a whole it "
                            "outranks both mileage and kilometres driven. Seller type is "
                            "the one field that barely registers. That roughly matches how "
                            "people judge a used car: how old it is and how powerful it is "
                            "jump out first, then what it burns and whose badge is on it."
                        ),
                    ],
                    className="card",
                ),
                html.Div(
                    [
                        html.H2("A few honest caveats"),
                        html.P(
                            "The model tells you what price fits the pattern in past "
                            "sales, not what a car \"should\" cost. It doesn't know about "
                            "condition, service history, accident damage, or how badly "
                            "someone wants to sell — all things that move a real "
                            "negotiation. Treat the number as a starting point, not the "
                            "final word."
                        ),
                        html.P(
                            "Accuracy is not even across the price range. Around 73% of "
                            "cars in testing were priced within 20% of what they actually "
                            "sold for, but that figure comes mostly from the ordinary "
                            "middle of the market. Expensive cars above roughly 2M are "
                            "thin in the training data and the model tends to underprice "
                            "them, so treat estimates on luxury vehicles with extra "
                            "scepticism."
                        ),
                        html.P(
                            "CNG and LPG cars aren't supported in the predictor either, "
                            "for the same reason they were excluded from training: their "
                            "mileage is reported in a different unit and mixing it in "
                            "would confuse the model."
                        ),
                    ],
                    className="card",
                ),
            ],
            className="content-stack",
        ),
    ],
    className="page",
)
