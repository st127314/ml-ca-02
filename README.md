# A2: Predicting Car Price — linear regression from scratch

Continues from [A1](https://github.com/st127314/ml-ca-01). Same dataset, same cleaning,
but the modelling is replaced with a `LinearRegression` written from scratch and extended
from the class notebook `03 - Bias-Variance Tradeoff and Regularization.ipynb`.

Three parts:

1. `app/code/linear_regression.py` — the class, with `r2`, Xavier initialisation, momentum
   and a coefficient-based feature importance plot added.
2. `notebooks/02_car_price_scratch_regression.ipynb` — 144 configurations compared under
   cross validation, all tracked in MLflow.
3. `app/` — the A1 web app with a second prediction page served by the new model.

## Where the model class lives, and why

In `app/code/linear_regression.py` rather than inside the notebook. Pickle stores the
*import path* of a class, not its code, so a class defined in a notebook is recorded as
living in `__main__` and the web app would fail to load the saved model. The notebook
imports the module and prints its full source, so the implementation is still readable in
one place for marking.

`app/code/data_prep.py` holds the cleaning for the same reason: the notebook and the app
cannot drift apart if there is only one definition of "the cleaned dataset".

## Task 1 — the class

Added on top of the class version:

- **`r2()`** — 1 − SS_res/SS_tot. Goes negative when a model does worse than predicting the
  mean, which turns out to be a useful signal: a learning rate of 0.0001 with batch gradient
  descent produces exactly that inside the epoch budget.
- **Xavier initialisation** — weights drawn from U[−1/√m, 1/√m], selectable against the
  original zeros. On a convex loss both land in the same place, which the notebook shows;
  the reason to implement it is the habit for deeper models.
- **Momentum** — with a selectable coefficient, see the note below.
- **`feature_importance()`** — ranks features by coefficient magnitude, keeping the sign in
  the bar colour. Only meaningful because every feature is standardised first; otherwise a
  coefficient reflects the unit its feature is measured in rather than its importance.

Two bugs in the class version are fixed, both commented where they occur: `val_loss_old`
was set once before the fold loop rather than per fold (so a low loss in fold 1 could trip
early stopping on fold 2's first epoch), and `prev_step` needs resetting per fold or
momentum carries velocity across folds.

### A discrepancy in the assignment's momentum pseudocode

The brief gives:

```
step  = alpha * grad
theta = theta - step + momentum * prev_step
prev_step = step
```

Read literally, `prev_step` holds `alpha * grad`, which points **uphill**, so adding it back
works against the descent. Measured, it converges *slower* than no momentum at all — the
opposite of what the brief's own prose describes:

| Variant | 200 epochs | 2000 epochs |
| --- | --- | --- |
| No momentum | R² 0.924 | R² 0.998 |
| Brief's pseudocode | R² −0.810 | R² 0.922 |
| Standard momentum | R² 0.998 | R² 0.998 |

If `prev_step` instead holds the update actually applied (`−step`), the same line is textbook
momentum. Both are implemented and selectable via `momentum_variant`; `standard` is the
default because it matches the described behaviour. The notebook shows the comparison on the
real data too.

## Task 2 — the experiment

| Axis | Values | n |
| --- | --- | --- |
| Model | normal, lasso, ridge, polynomial | 4 |
| Momentum | without, with | 2 |
| Gradient descent | batch, mini-batch, stochastic | 3 |
| Initialisation | zeros, xavier | 2 |
| Learning rate | 0.01, 0.001, 0.0001 | 3 |

144 configurations, 3-fold cross validation, 432 fits.

**On the epoch budget.** An epoch means something different per method: batch does one
weight update per epoch, mini-batch about 109, stochastic 5,444. Giving all three the same
number of epochs would compare how many updates each was allowed, not the methods. Measured
first: batch at 500 epochs reaches R² −0.07 and needs around 5,000 to converge, while
stochastic is already at 0.85 after 5. The budget equalises updates roughly and is then held
fixed across every configuration.

## MLflow

Local SQLite backend by default, written to `mlflow.db`. MLflow 3 retired the plain
`./mlruns` file store, and SQLite is the recommended replacement — still one local file with
no server to run.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`.

To use a remote tracking server instead, export `MLFLOW_TRACKING_URI` before starting
Jupyter; the notebook picks it up and needs no other change.

The class logs the loss curve every 50 epochs rather than every epoch. Per-epoch logging
across the full grid wrote 1.4 million metric rows and a 350 MB tracking database; at every
50 it is 27,000 rows and the curves look the same.

## Running it

```bash
pip install -r app/requirements.txt
pip install mlflow pytest

# tests: the class is checked on a synthetic problem where the answer is known
python -m pytest

# the experiment
jupyter lab notebooks/02_car_price_scratch_regression.ipynb
```

## The web app

```bash
python app/code/app.py            # http://127.0.0.1:8050
docker compose -f app/docker-compose.yaml up --build
```

Three pages:

- `/` — the landing page
- `/predict` — the A1 Random Forest
- `/predict-scratch` — the new from-scratch model, with an explanation of how the two differ
  and the test-set numbers behind the claim

## Deployment

Deployed on the class server behind Traefik, which terminates TLS and routes by subdomain,
so the container publishes no ports of its own.
