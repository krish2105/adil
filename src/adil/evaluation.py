"""Discrimination and calibration metrics, and the intervals around them.

Holds: the two metrics ADIL computes itself, the assembled metric set every
notebook reports, and a bootstrap interval.

Most of what this project measures comes from :mod:`spine.metrics`, which
verifies itself against scikit-learn and the standard library in
``reports/proof.json``. Only KS and Gini are written here, because SPINE does not
carry them and they are three lines each. They are checked against a four-row
case that can be worked on paper.

Two conventions, both from the project's evaluation rules:

Every metric is reported alongside the split it was computed on. A metric set
cannot be constructed without naming one, which is why ``split`` is a required
argument rather than a label added afterwards.

PR-AUC leads and AUC follows. At an ~8% base rate AUC flatters: a model can rank
well and still be useless at the low-prevalence end where the approval threshold
actually sits. AUC is reported because a credit audience expects Gini, which is
derived from it, not because it is the informative number.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score, roc_curve
from spine.metrics import brier_score, expected_calibration_error, pr_auc

__all__ = ["bootstrap_interval", "gini", "ks_statistic", "metric_set"]


def ks_statistic(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Kolmogorov-Smirnov separation between good and bad accounts.

    The largest vertical gap between the two cumulative score distributions,
    equivalently ``max(TPR - FPR)`` over the ROC curve. Long-standing in credit
    because it names the cut where the two populations are furthest apart —
    though that cut is rarely the one a cost matrix chooses, which is why
    notebook 06 does not use it to set the threshold.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes; 1 is default.
    y_score : array-like of shape (n,)
        Scores or probabilities. Higher must mean more likely to default.

    Returns
    -------
    float
        Between 0 and 1.

    Examples
    --------
    Perfect separation.

    >>> ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
    1.0

    One swap in the middle costs half of it.

    >>> ks_statistic([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])
    0.5
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    return float(np.max(true_positive_rate - false_positive_rate))


def gini(y_true: ArrayLike, y_score: ArrayLike) -> float:
    r"""Gini coefficient, the credit industry's rescaling of AUC.

    .. math:: G = 2\,\\mathrm{AUC} - 1

    Reported because a credit audience reads Gini, not because it says anything
    AUC does not.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes; 1 is default.
    y_score : array-like of shape (n,)
        Scores or probabilities.

    Returns
    -------
    float
        Between -1 and 1. Zero is a coin toss.

    Examples
    --------
    >>> gini([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
    1.0
    >>> gini([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4])
    -1.0
    """
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def metric_set(y_true: ArrayLike, y_prob: ArrayLike, split: str) -> dict[str, object]:
    """Assemble every metric the headline table reports, for one split.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes; 1 is default.
    y_prob : array-like of shape (n,)
        Predicted probabilities of default. Brier and ECE are meaningless on an
        unscaled score, so pass probabilities rather than scorecard points.
    split : str
        Which split these were computed on. Required, not optional: a metric
        without its split is not reportable.

    Returns
    -------
    dict
        Plain Python scalars, ready for ``json.dump``.

    Examples
    --------
    >>> result = metric_set([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], split="test")
    >>> result["split"], result["n"], result["auc"]
    ('test', 4, 1.0)
    """
    truth = np.asarray(y_true)
    probability = np.asarray(y_prob, dtype=float)
    return {
        "split": split,
        "n": int(truth.size),
        "event_rate": float(truth.mean()),
        "pr_auc": float(pr_auc(truth, probability)),
        "auc": float(roc_auc_score(truth, probability)),
        "ks": ks_statistic(truth, probability),
        "gini": gini(truth, probability),
        "brier": float(brier_score(truth, probability)),
        "ece": float(expected_calibration_error(truth, probability)),
    }


def bootstrap_interval(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    metric: Callable[[np.ndarray, np.ndarray], float],
    draws: int = 400,
    level: float = 0.95,
    seed: int = 20260827,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric on a fixed model.

    Resamples predictions rather than refitting, so this is the sampling
    uncertainty of the *measurement* on this test set, not the uncertainty of the
    model. It answers "how precisely do I know this model's AUC here", not "how
    much would this AUC move on different training data" — the second is the
    larger uncertainty and is not estimated anywhere in this project.

    The project's evaluation rules ask for intervals rather than bare point
    estimates. A classifier produces no posterior, so this is the honest
    substitute and the limitation above is stated wherever it is reported.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes.
    y_prob : array-like of shape (n,)
        Predicted probabilities.
    metric : callable
        Takes ``(y_true, y_prob)`` and returns a float.
    draws : int
        Bootstrap resamples.
    level : float
        Coverage, so 0.95 gives the 2.5th and 97.5th percentiles.
    seed : int
        Recorded alongside the interval.

    Returns
    -------
    low, high : float

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.binomial(1, 0.5, 200)
    >>> p = np.where(y == 1, 0.7, 0.3)
    >>> low, high = bootstrap_interval(y, p, gini, draws=50)
    >>> low <= gini(y, p) <= high
    True
    """
    truth = np.asarray(y_true)
    probability = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        index = rng.integers(0, truth.size, truth.size)
        resampled = truth[index]
        # A resample that lost one class entirely cannot support a ranking metric.
        if resampled.min() == resampled.max():
            continue
        values.append(metric(resampled, probability[index]))
    tail = (1 - level) / 2
    return (
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1 - tail)),
    )
