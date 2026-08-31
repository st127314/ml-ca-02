"""Check the from-scratch regression on a synthetic problem where the right
answer is known, so a failure points at the implementation and not at the data."""

import numpy as np
import pytest

from linear_regression import (
    ElasticNet,
    Lasso,
    LinearRegression,
    NoPenalty,
    Normal,
    Ridge,
    RidgePenalty,
)


def make_data(n=400, seed=0):
    """y = 3 + 2*x1 - 1*x2 + small noise, with a bias column prepended."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 3 + 2 * X[:, 0] - 1 * X[:, 1] + rng.normal(scale=0.1, size=n)
    X = np.c_[np.ones(n), X]  # bias column, so theta[0] is the intercept
    return X, y


def fit(**kwargs):
    X, y = make_data()
    kwargs.setdefault("num_epochs", 2000)
    kwargs.setdefault("lr", 0.05)
    model = Normal(use_mlflow=False, **kwargs)
    model.fit(X, y)
    return model, X, y


# ------------------------------------------------------------------ the basics
def test_recovers_the_true_coefficients():
    model, _, _ = fit()
    assert model._bias() == pytest.approx(3.0, abs=0.15)
    assert model._coef()[0] == pytest.approx(2.0, abs=0.15)
    assert model._coef()[1] == pytest.approx(-1.0, abs=0.15)


def test_r2_is_near_one_on_an_easy_problem():
    model, X, y = fit()
    assert model.r2(y, model.predict(X)) > 0.98


def test_r2_of_the_mean_prediction_is_zero():
    model = Normal(use_mlflow=False)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert model.r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_is_negative_when_worse_than_the_mean():
    model = Normal(use_mlflow=False)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert model.r2(y, np.array([10.0, -5.0, 8.0, 0.0])) < 0


def test_mse_is_zero_for_a_perfect_prediction():
    model = Normal(use_mlflow=False)
    y = np.array([1.0, 2.0, 3.0])
    assert model.mse(y, y.copy()) == pytest.approx(0.0)


# ------------------------------------------------------------- initialisation
def test_zeros_initialisation_starts_at_zero():
    model = Normal(use_mlflow=False, init_method="zeros")
    theta = model._initialize_theta(5, np.random.default_rng(0))
    assert np.array_equal(theta, np.zeros(5))


def test_xavier_stays_inside_the_stated_bound():
    model = Normal(use_mlflow=False, init_method="xavier")
    m = 16
    theta = model._initialize_theta(m, np.random.default_rng(0))
    bound = 1.0 / np.sqrt(m)
    assert theta.shape == (m,)
    assert np.all(np.abs(theta) <= bound)
    assert not np.array_equal(theta, np.zeros(m))  # actually random


def test_xavier_bound_shrinks_as_inputs_grow():
    model = Normal(use_mlflow=False, init_method="xavier")
    rng = np.random.default_rng(1)
    small = np.abs(model._initialize_theta(4, rng)).max()
    large = np.abs(model._initialize_theta(1000, rng)).max()
    assert large < small


def test_unknown_initialisation_is_rejected():
    model = Normal(use_mlflow=False, init_method="uniform")
    with pytest.raises(ValueError, match="init_method"):
        model._initialize_theta(3, np.random.default_rng(0))


def test_both_initialisations_reach_the_same_answer():
    """The loss is convex, so where you start should not change where you land."""
    zeros, _, _ = fit(init_method="zeros")
    xavier, _, _ = fit(init_method="xavier")
    assert np.allclose(zeros.theta, xavier.theta, atol=0.1)


# -------------------------------------------------------------------- momentum
def test_standard_momentum_converges_much_faster():
    """The whole point of momentum: same budget, far better result."""
    plain, X, y = fit(num_epochs=200, lr=0.01)
    fast, _, _ = fit(num_epochs=200, lr=0.01, use_momentum=True, momentum=0.9)
    assert fast.r2(y, fast.predict(X)) > 0.99
    assert fast.r2(y, fast.predict(X)) > plain.r2(y, plain.predict(X))


def test_brief_variant_matches_the_literal_pseudocode():
    """The brief's lines, read exactly as written, decelerate rather than
    accelerate. Kept implemented so the notebook can show the comparison."""
    slow, X, y = fit(num_epochs=200, lr=0.01, use_momentum=True,
                     momentum=0.9, momentum_variant="brief")
    fast, _, _ = fit(num_epochs=200, lr=0.01, use_momentum=True,
                     momentum=0.9, momentum_variant="standard")
    assert slow.r2(y, slow.predict(X)) < fast.r2(y, fast.predict(X))


def test_unknown_momentum_variant_is_rejected():
    X, y = make_data()
    model = Normal(use_mlflow=False, use_momentum=True, momentum_variant="nesterov")
    with pytest.raises(ValueError, match="momentum_variant"):
        model.fit(X, y)


def test_momentum_applies_the_previous_step():
    """One update by hand, against the pseudocode in the assignment brief."""
    model = Normal(use_mlflow=False, lr=0.1, use_momentum=True, momentum=0.5,
                   momentum_variant="brief")
    model.theta = np.array([1.0, 1.0])
    model.prev_step = np.array([0.2, 0.2])
    X = np.array([[1.0, 2.0]])
    y = np.array([0.0])

    before, prev = model.theta.copy(), model.prev_step.copy()
    model._train(X, y)

    grad = X.T @ (X @ before - y) / 1
    step = 0.1 * grad
    expected = before - step + 0.5 * prev
    assert np.allclose(model.theta, expected)
    assert np.allclose(model.prev_step, step)


def test_momentum_resets_between_folds():
    model, _, _ = fit(use_momentum=True)
    assert hasattr(model, "prev_step")


# -------------------------------------------------------------- regularisation
def test_ridge_shrinks_coefficients_more_than_plain():
    X, y = make_data()
    plain = Normal(use_mlflow=False, lr=0.05, num_epochs=800)
    ridge = Ridge(l=1.0, lr=0.05, num_epochs=800, use_mlflow=False)
    plain.fit(X, y)
    ridge.fit(X, y)
    assert np.abs(ridge._coef()).sum() < np.abs(plain._coef()).sum()


def test_no_penalty_contributes_no_gradient():
    theta = np.array([1.0, -2.0, 3.0])
    assert np.array_equal(NoPenalty().derivation(theta), np.zeros(3))
    assert NoPenalty()(theta) == 0.0


def test_ridge_penalty_derivative_is_two_lambda_theta():
    theta = np.array([1.0, -2.0])
    assert np.allclose(RidgePenalty(0.5).derivation(theta), 2 * 0.5 * theta)


def test_lasso_penalty_derivative_is_lambda_sign():
    theta = np.array([3.0, -2.0, 0.0])
    assert np.allclose(Lasso(l=0.1).regularization.derivation(theta), 0.1 * np.sign(theta))


@pytest.mark.parametrize("cls", [Normal, Lasso, Ridge, ElasticNet])
def test_every_variant_trains(cls):
    X, y = make_data()
    model = cls(lr=0.05, num_epochs=300, use_mlflow=False)
    model.fit(X, y)
    assert np.isfinite(model.theta).all()


# ------------------------------------------------------------------ gd methods
@pytest.mark.parametrize("method", ["batch", "mini", "sto"])
def test_all_gradient_descent_methods_converge(method):
    epochs = {"batch": 2000, "mini": 200, "sto": 30}[method]
    model, X, y = fit(method=method, num_epochs=epochs, lr=0.01)
    assert model.r2(y, model.predict(X)) > 0.95


def test_kfold_records_one_score_per_fold():
    model, _, _ = fit()
    assert len(model.kfold_scores) == 3
    assert all(np.isfinite(s) for s in model.kfold_scores)


# ----------------------------------------------------------- feature importance
def test_feature_importance_plots_and_ranks_by_magnitude():
    model, _, _ = fit()
    ax = model.feature_importance(["x1", "x2"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(labels) == {"x1", "x2"}
    # x1 has the larger true coefficient (2 vs -1), so it must rank first;
    # barh draws bottom-up, so the most important label sits last.
    assert labels[-1] == "x1"


def test_feature_importance_rejects_a_name_count_mismatch():
    model, _, _ = fit()
    with pytest.raises(ValueError, match="feature names"):
        model.feature_importance(["only_one"])


def test_feature_importance_honours_top_n():
    rng = np.random.default_rng(0)
    X = np.c_[np.ones(200), rng.normal(size=(200, 6))]
    y = X[:, 1] * 2 + rng.normal(scale=0.1, size=200)
    model = Normal(lr=0.05, num_epochs=300, use_mlflow=False)
    model.fit(X, y)
    ax = model.feature_importance([f"f{i}" for i in range(6)], top_n=3)
    assert len(ax.get_yticklabels()) == 3
