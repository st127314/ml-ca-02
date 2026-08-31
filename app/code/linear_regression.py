"""Linear regression from scratch, extending the class version in
`03 - Bias-Variance Tradeoff and Regularization.ipynb`.

This lives in a module rather than only inside the notebook on purpose. The trained
model is pickled and loaded again by the Dash app, and pickle stores the *import path*
of a class, not its code. A class defined in a notebook is recorded as living in
`__main__`, so the app would fail to unpickle it. Importing from here means the
notebook and the web app agree on where the class lives.

Additions on top of the class version, all required by A2 Task 1:
  * `r2()`                    - the R^2 score
  * Xavier weight initialisation, selectable against the original zeros
  * momentum, with a selectable coefficient
  * `feature_importance()`    - plot importance from the fitted coefficients
"""

import numpy as np
from sklearn.model_selection import KFold

try:  # mlflow is required for the A2 experiment but not for importing the class,
    import mlflow  # so the app can load a saved model without mlflow installed.
except ImportError:  # pragma: no cover
    mlflow = None


# ---------------------------------------------------------------------------
# Penalties. Each one exposes __call__ (the penalty value, for reporting) and
# derivation (the gradient contribution, which is what _train actually uses).
# ---------------------------------------------------------------------------
class NoPenalty:
    """Plain linear regression: no penalty, so no gradient contribution."""

    def __init__(self, l=0):
        self.l = l

    def __call__(self, theta):
        return 0.0

    def derivation(self, theta):
        return np.zeros_like(theta)


class LassoPenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)


class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class ElasticPenalty:
    def __init__(self, l=0.1, l_ratio=0.5):
        self.l = l
        self.l_ratio = l_ratio

    def __call__(self, theta):
        l1 = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2 = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return l1 + l2

    def derivation(self, theta):
        l1 = self.l * self.l_ratio * np.sign(theta)
        l2 = self.l * (1 - self.l_ratio) * theta
        return l1 + l2


class LinearRegression:
    """Gradient-descent linear regression with k-fold cross validation built in.

    Parameters mirror the class notebook, plus the three A2 additions
    (`init_method`, `use_momentum`, `momentum`).

    method : 'batch' uses the whole fold each step, 'mini' uses `batch_size` rows,
             'sto' uses a single row. They trade gradient quality against how many
             updates you get per pass over the data.
    """

    kfold = KFold(n_splits=3)

    def __init__(
        self,
        regularization,
        lr=0.001,
        method="batch",
        num_epochs=500,
        batch_size=50,
        cv=kfold,
        init_method="zeros",
        use_momentum=False,
        momentum=0.9,
        momentum_variant="standard",
        use_mlflow=True,
        log_every=1,
        random_state=42,
    ):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method
        self.cv = cv
        self.regularization = regularization
        self.init_method = init_method
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.momentum_variant = momentum_variant
        self.use_mlflow = use_mlflow and mlflow is not None
        # Log the loss curve every `log_every` epochs rather than every single one.
        # Batch gradient descent needs thousands of epochs to converge, and logging
        # two metrics per epoch across the whole grid produced 1.4M rows and a 350MB
        # tracking database - almost all of it redundant for drawing a loss curve.
        self.log_every = max(1, int(log_every))
        self.random_state = random_state

    # ------------------------------------------------------------------ metrics
    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        """R^2 = 1 - SS_res / SS_tot.

        SS_tot is the error a model that only ever predicts the mean would make,
        so R^2 is 'how much of the variance did we explain over that baseline'.
        1.0 is perfect, 0.0 is no better than the mean, negative is worse.
        """
        ss_res = ((ytrue - ypred) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot)

    # ------------------------------------------------------- weight initialising
    def _initialize_theta(self, n_features, rng):
        """Zeros (the class default) or Xavier uniform.

        Xavier draws from U[-1/sqrt(m), 1/sqrt(m)] where m is the number of inputs.
        The point is to keep the starting signal at a sensible magnitude: too large
        and early gradients are huge, too small and they vanish. Zeros are a fine
        starting point for plain linear regression because the loss is convex, but
        Xavier is the habit worth building for deeper models.
        """
        if self.init_method == "zeros":
            return np.zeros(n_features)
        if self.init_method == "xavier":
            bound = 1.0 / np.sqrt(n_features)
            return rng.uniform(low=-bound, high=bound, size=n_features)
        raise ValueError(f"init_method must be 'zeros' or 'xavier', got {self.init_method!r}")

    # ---------------------------------------------------------------- training
    def fit(self, X_train, y_train):
        # mse per fold (what the class version kept) and r2 per fold, because A2
        # asks for the comparison in terms of both.
        self.kfold_scores = list()
        self.kfold_r2_scores = list()
        rng = np.random.default_rng(self.random_state)

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            self.theta = self._initialize_theta(X_cross_train.shape[1], rng)
            # Momentum carries the previous step into the next update, so it has to
            # start empty for every fold, otherwise fold 2 inherits fold 1's velocity.
            self.prev_step = np.zeros_like(self.theta)
            # Reset per fold as well: the class version sets this once before the
            # loop, which lets a low validation loss from an earlier fold trip the
            # early-stopping check on the very first epoch of the next one.
            self.val_loss_old = np.inf

            run_ctx = (
                mlflow.start_run(run_name=f"Fold-{fold}", nested=True)
                if self.use_mlflow
                else _NullRun()
            )
            with run_ctx:
                if self.use_mlflow:
                    mlflow.log_params(
                        {
                            "method": self.method,
                            "lr": self.lr,
                            "reg": type(self).__name__,
                            "init": self.init_method,
                            "momentum": self.momentum if self.use_momentum else 0,
                            "momentum_variant": self.momentum_variant if self.use_momentum else "none",
                        }
                    )

                for epoch in range(self.num_epochs):
                    perm = rng.permutation(X_cross_train.shape[0])
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]

                    if self.method == "sto":
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1)
                            y_method_train = y_cross_train[batch_idx].reshape(1)
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == "mini":
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            X_method_train = X_cross_train[batch_idx : batch_idx + self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx : batch_idx + self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        train_loss = self._train(X_cross_train, y_cross_train)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)

                    if self.use_mlflow and (
                        epoch % self.log_every == 0 or epoch == self.num_epochs - 1
                    ):
                        mlflow.log_metric("train_loss", train_loss, step=epoch)
                        mlflow.log_metric("val_loss", val_loss_new, step=epoch)

                    # Stop once the validation loss stops moving: more epochs after
                    # that only cost time and risk drifting into overfitting.
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new

                self.kfold_scores.append(val_loss_new)
                self.kfold_r2_scores.append(self.r2(y_cross_val, self.predict(X_cross_val)))

        self.avg_val_mse = float(np.mean(self.kfold_scores))
        self.avg_val_r2 = float(np.mean(self.kfold_r2_scores))
        return self

    def _train(self, X, y):
        yhat = self.predict(X)
        m = X.shape[0]
        grad = (1 / m) * X.T @ (yhat - y) + self.regularization.derivation(self.theta)

        # Momentum keeps a fraction of the previous move. Directions that stay
        # consistent build up speed; directions that keep flipping cancel out,
        # which is what damps oscillation across a narrow valley.
        #
        # Two readings of the brief's pseudocode, kept switchable because they
        # behave very differently (see the momentum section of the notebook):
        #
        #   'brief'    - literal transcription. prev_step holds alpha*grad, which
        #                points uphill, so adding it back FIGHTS the descent. On the
        #                synthetic check this converged slower than no momentum at
        #                all (R2 0.92 vs 0.998 after 2000 epochs).
        #   'standard' - prev_step holds the update actually applied. This is
        #                textbook momentum and matches what the brief's prose
        #                describes: faster convergence, reduced oscillation
        #                (R2 0.998 in 200 epochs instead of 2000).
        if not self.use_momentum:
            step = self.lr * grad
            self.theta = self.theta - step
            self.prev_step = step
        elif self.momentum_variant == "brief":
            step = self.lr * grad
            self.theta = self.theta - step + self.momentum * self.prev_step
            self.prev_step = step
        elif self.momentum_variant == "standard":
            step = self.lr * grad + self.momentum * self.prev_step
            self.theta = self.theta - step
            self.prev_step = step
        else:
            raise ValueError(
                f"momentum_variant must be 'standard' or 'brief', got {self.momentum_variant!r}"
            )

        return self.mse(y, yhat)

    def predict(self, X):
        return X @ self.theta

    # ---------------------------------------------------------------- inspection
    def _coef(self):
        return self.theta[1:]  # theta[0] is the bias / intercept

    def _bias(self):
        return self.theta[0]

    def feature_importance(self, feature_names, top_n=20, ax=None):
        """Plot importance as the magnitude of each fitted coefficient.

        This is only meaningful because every feature was standardised before
        fitting. Without that, a coefficient's size reflects the unit its feature
        happens to be measured in (km vs seats) rather than how much the feature
        matters, and the ranking would be meaningless.

        Sign is kept in the bar colour: magnitude ranks the features, direction
        says whether the feature pushes the predicted price up or down.
        """
        coef = np.asarray(self._coef())
        names = np.asarray(feature_names)
        if len(names) != len(coef):
            raise ValueError(f"got {len(names)} feature names for {len(coef)} coefficients")

        # Imported here rather than at module level: the web app imports this module
        # to unpickle the model and never plots, so keeping matplotlib out of the
        # import path keeps it out of app/requirements.txt and out of the image.
        import matplotlib.pyplot as plt

        order = np.argsort(np.abs(coef))[::-1][:top_n]
        coef, names = coef[order][::-1], names[order][::-1]

        if ax is None:
            _, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(coef))))
        ax.barh(names, coef, color=["#c0392b" if c < 0 else "#2471a3" for c in coef])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (standardised features)")
        ax.set_title(f"Top {len(coef)} features by coefficient magnitude")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return ax


class _NullRun:
    """Stand-in for an mlflow run so `fit` works with mlflow switched off."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Convenience subclasses, matching the class notebook's naming.
# ---------------------------------------------------------------------------
class Normal(LinearRegression):
    def __init__(self, method="batch", lr=0.001, l=0, **kwargs):
        super().__init__(NoPenalty(l), lr=lr, method=method, **kwargs)


class Lasso(LinearRegression):
    def __init__(self, method="batch", lr=0.001, l=0.01, **kwargs):
        super().__init__(LassoPenalty(l), lr=lr, method=method, **kwargs)


class Ridge(LinearRegression):
    def __init__(self, method="batch", lr=0.001, l=0.01, **kwargs):
        super().__init__(RidgePenalty(l), lr=lr, method=method, **kwargs)


class ElasticNet(LinearRegression):
    def __init__(self, method="batch", lr=0.001, l=0.01, l_ratio=0.5, **kwargs):
        super().__init__(ElasticPenalty(l, l_ratio), lr=lr, method=method, **kwargs)
