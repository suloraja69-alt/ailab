from __future__ import annotations
import math
from typing import Tuple, Dict, Any
import numpy as np


class NaiveBayesGaussian:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_prior_log_: Dict[Any, float] = {}
        self.mean_: Dict[Any, np.ndarray] = {}
        self.var_: Dict[Any, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesGaussian":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        for c in self.classes_:
            Xc = X[y == c]

            self.class_prior_log_[c] = math.log(len(Xc) / len(X))

            mu = Xc.mean(axis=0)
            var = Xc.var(axis=0) + self.var_smoothing

            self.mean_[c] = mu
            self.var_[c] = var

        return self

    def _log_gaussian_likelihood(
        self, x: np.ndarray, c: Any
    ) -> float:

        mu = self.mean_[c]
        var = self.var_[c]

        return (
            -0.5 * (np.log(2 * np.pi * var)).sum()
            - 0.5 * (((x - mu) ** 2) / var).sum()
        )

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        scores = []

        for x in X:
            row = []

            for c in self.classes_:
                row.append(
                    self.class_prior_log_[c]
                    + self._log_gaussian_likelihood(x, c)
                )

            scores.append(row)

        return np.array(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        jll = self._joint_log_likelihood(X)
        idx = jll.argmax(axis=1)

        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(
            np.asarray(X, dtype=float)
        )

        jll -= jll.max(axis=1, keepdims=True)

        probs = np.exp(jll)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs


class NaiveBayesMultinomial:
    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("alpha must be >=0")

        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.class_prior_log_: Dict[Any, float] = {}
        self.feature_log_prob_: Dict[Any, np.ndarray] = {}

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "NaiveBayesMultinomial":

        X = np.asarray(X, dtype=float)

        if (X < 0).any():
            raise ValueError(
                "Multinomial NB requires non-negative counts."
            )

        y = np.asarray(y)
        self.classes_ = np.unique(y)

        for c in self.classes_:
            Xc = X[y == c]

            self.class_prior_log_[c] = math.log(
                len(Xc) / len(X)
            )

            counts = Xc.sum(axis=0)
            total = counts.sum()

            probs = (
                counts + self.alpha
            ) / (
                total + self.alpha * X.shape[1]
            )

            self.feature_log_prob_[c] = np.log(probs)

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        scores = []

        for x in X:
            row = []

            for c in self.classes_:
                row.append(
                    self.class_prior_log_[c]
                    + float(
                        np.dot(
                            x,
                            self.feature_log_prob_[c]
                        )
                    )
                )

            scores.append(row)

        return np.array(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        jll = self._joint_log_likelihood(X)
        idx = jll.argmax(axis=1)

        return self.classes_[idx]


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return (y_true == y_pred).mean()


def _make_gaussian_blob_data(
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    mean0 = np.array([0.0, 0.0])
    cov0 = np.array([
        [1.0, 0.4],
        [0.4, 1.2]
    ])

    mean1 = np.array([2.5, 2.0])
    cov1 = np.array([
        [1.1, -0.3],
        [-0.3, 1.0]
    ])

    X0 = rng.multivariate_normal(
        mean0, cov0, size=120
    )

    X1 = rng.multivariate_normal(
        mean1, cov1, size=120
    )

    y0 = np.zeros(len(X0), dtype=int)
    y1 = np.ones(len(X1), dtype=int)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    idx = rng.permutation(len(X))

    split = int(0.75 * len(X))

    train_idx = idx[:split]
    test_idx = idx[split:]

    return (
        X[train_idx],
        y[train_idx],
        X[test_idx],
        y[test_idx]
    )


def _make_multinomial_spam_data():

    vocab = [
        "buy",
        "discount",
        "meeting",
        "project"
    ]

    X = np.array([
        [2, 1, 0, 0],
        [1, 2, 0, 0],
        [0, 0, 2, 1],
        [0, 0, 1, 2],
        [3, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 3],
        [1, 0, 1, 0],
    ], dtype=float)

    y = np.array(
        [1, 1, 0, 0, 1, 0, 0, 0],
        dtype=int
    )

    train_idx = np.array([0, 1, 2, 3, 4, 5])
    test_idx = np.array([6, 7])

    return (
        X[train_idx],
        y[train_idx],
        X[test_idx],
        y[test_idx],
        vocab
    )


if __name__ == "__main__":

    Xtr_g, ytr_g, Xte_g, yte_g = \
        _make_gaussian_blob_data(seed=7)

    gnb = NaiveBayesGaussian(
        var_smoothing=1e-9
    ).fit(Xtr_g, ytr_g)

    yhat_g = gnb.predict(Xte_g)

    acc_g = accuracy(yte_g, yhat_g)

    print(
        f"[GaussianNB] Test accuracy: {acc_g:.3f}"
    )

    print(
        "Forward-looking note: suitable for continuous "
        "features with class-conditional normality.\n"
    )

    Xtr_m, ytr_m, Xte_m, yte_m, vocab = \
        _make_multinomial_spam_data()

    mnb = NaiveBayesMultinomial(
        alpha=1.0
    ).fit(Xtr_m, ytr_m)

    yhat_m = mnb.predict(Xte_m)

    acc_m = accuracy(yte_m, yhat_m)

    print(
        f"[MultinomialNB] Test accuracy: {acc_m:.3f}"
    )

    print(f"Vocabulary: {vocab}\n")

    if acc_g >= 0.5 and acc_m >= 0.5:
        print(
            "Thus, both Naive Bayes models were trained, "
            "evaluated, and executed successfully."
        )
    else:
        print(
            "Thus, the implementation executed, but model "
            "accuracy is low; consider revisiting data or smoothing."
        )
