r"""The cost model that turns a probability into an approval decision.

Holds: the 2×2 cost matrix, the closed-form threshold it implies, the accounting
of what a given cutoff actually costs, and the labelled scenario that puts those
costs in dirhams.

Two things about this module are unusual, and both are about honesty rather than
technique.

**The cost ratio is not invented.** Home Credit publishes no cost of default and
no margin, so any specific figure would be a number typed into a notebook and
then reported as though it meant something. The one published ratio available
anywhere in this project's data is UCI German Credit's documented matrix — five
to one against classing a bad account as good. It is used as the reference point
and the whole curve is reported around it, so no single assumed ratio drives the
conclusion.

**No amount here is in dirhams.** Home Credit's currency is anonymised and
unscaled, so "cost in AED per approved application" is not computable from this
data at all. Primary results are in dataset currency units. A single illustrative
AED scenario is reported alongside, carrying :data:`SCENARIO_CAVEAT` at every
output, with its assumptions stated where they can be checked.

Sign convention follows :mod:`spine.decisions`, which indexes ``[true][predicted]``
and predicts the *event*. The event is default, so predicting it means declining.

.. math::

    C = \begin{pmatrix} 0 & c_o \\ c_u & 0 \end{pmatrix}

where :math:`c_o` is the margin forgone by declining an applicant who would have
repaid, and :math:`c_u` is the loss on approving one who defaults.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AED_SCENARIO",
    "GERMAN_CREDIT_COST_RATIO",
    "GERMAN_CREDIT_SOURCE",
    "RATIO_GRID",
    "SCENARIO_CAVEAT",
    "analytic_threshold",
    "cost_matrix",
    "cost_matrix_from_amounts",
    "decision_costs",
]

#: Cost of approving a defaulter relative to declining a good applicant, as
#: documented in the UCI German Credit data. The only ratio in this project that
#: was published by someone else rather than assumed by its author.
GERMAN_CREDIT_COST_RATIO = 5.0

GERMAN_CREDIT_SOURCE = (
    "UCI Statlog German Credit Data, german.doc section 8 'Cost Matrix': "
    "'It is worse to class a customer as good when they are bad (5), than it is to "
    "class a customer as bad when they are good (1).' Reproduced as a ratio only — "
    "German Credit's amounts are in Deutsche Marks and do not transfer to Home Credit."
)

#: Ratios the expected-cost curve is reported across, so the conclusion does not
#: rest on one assumed figure.
RATIO_GRID: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)

SCENARIO_CAVEAT = (
    "SCENARIO — assumed parameters, not a finding. Home Credit's currency is anonymised "
    "and unscaled, so no dirham figure is computable from this data. The amounts below "
    "apply stated assumptions to the model's decisions to make the scale legible; they "
    "are not a measurement of anything and must not be quoted as one."
)

#: Assumptions behind the illustrative dirham figures. Every one is a choice, and
#: every one is written down here so it can be argued with.
AED_SCENARIO: dict[str, object] = {
    "assumed_mean_exposure_aed": 60_000.0,
    "assumed_loss_given_default": 0.45,
    "assumed_lifetime_margin_rate": 0.09,
    "basis": (
        "A 60,000 AED mean personal loan, 45% loss given default, and a 9% lifetime "
        "margin on an approved loan that performs. These are round illustrative figures "
        "chosen to be arguable, not estimates drawn from any UAE portfolio."
    ),
    "caveat": SCENARIO_CAVEAT,
}


def cost_matrix(ratio: float = GERMAN_CREDIT_COST_RATIO) -> NDArray[np.float64]:
    """Build the 2×2 cost matrix from the loss-to-margin ratio.

    Only the ratio can move a threshold. Scaling both costs by the same factor
    scales the total cost and leaves the cutoff exactly where it was, which is
    why the reference point here is a ratio rather than a pair of amounts.

    Parameters
    ----------
    ratio : float
        Cost of approving a defaulter, expressed in units of the margin forgone
        by declining an applicant who would have repaid.

    Returns
    -------
    numpy.ndarray of shape (2, 2)
        Indexed ``[true label][predicted label]``, ready for
        :func:`spine.decisions.optimal_threshold`.

    Raises
    ------
    ValueError
        If the ratio is not positive.

    Examples
    --------
    >>> cost_matrix(5.0).tolist()
    [[0.0, 1.0], [5.0, 0.0]]
    """
    if ratio <= 0:
        raise ValueError(f"the cost ratio must be positive; got {ratio}")
    return np.array([[0.0, 1.0], [float(ratio), 0.0]])


def cost_matrix_from_amounts(loss: float, margin: float) -> NDArray[np.float64]:
    """Build the cost matrix from two currency amounts.

    Parameters
    ----------
    loss : float
        Expected loss on approving an applicant who defaults.
    margin : float
        Margin forgone by declining one who would have repaid.

    Returns
    -------
    numpy.ndarray of shape (2, 2)
        In the same currency units as the inputs.

    Raises
    ------
    ValueError
        If either amount is not positive.

    Examples
    --------
    >>> cost_matrix_from_amounts(loss=4500.0, margin=900.0).tolist()
    [[0.0, 900.0], [4500.0, 0.0]]
    """
    if loss <= 0 or margin <= 0:
        raise ValueError(f"both amounts must be positive; got loss={loss}, margin={margin}")
    return np.array([[0.0, float(margin)], [float(loss), 0.0]])


def analytic_threshold(matrix: ArrayLike) -> float:
    r"""Cost-minimising cutoff for a perfectly calibrated model.

    .. math:: t^* = \frac{c_o}{c_o + c_u}

    Where this disagrees with the threshold found by searching the observed
    probabilities, the disagreement measures miscalibration — and the analytic
    value is the one that is wrong, because it assumes a calibration the model
    may not have. Notebook 06 reports the gap for exactly that reason.

    Parameters
    ----------
    matrix : array-like of shape (2, 2)
        Cost matrix indexed ``[true][predicted]``.

    Returns
    -------
    float
        Decline when the predicted default probability is at or above this.

    Examples
    --------
    >>> round(analytic_threshold(cost_matrix(5.0)), 6)
    0.166667
    """
    costs = np.asarray(matrix, dtype=float)
    forgone_margin = costs[0, 1] - costs[0, 0]
    loss = costs[1, 0] - costs[1, 1]
    return float(forgone_margin / (forgone_margin + loss))


def decision_costs(
    y_true: ArrayLike, y_prob: ArrayLike, threshold: float, cost_matrix: ArrayLike
) -> dict[str, float]:
    """Account for what one cutoff costs, and on what book.

    Applications are approved when the predicted default probability is *below*
    the threshold, matching :mod:`spine.decisions`, where predicting the event
    means declining.

    Both a per-application and a per-approved-application cost are reported. The
    second is the one the research question asks for and the one a lender reads,
    because a cutoff that declines almost everybody has an excellent cost per
    application and no business.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcomes; 1 is default.
    y_prob : array-like of shape (n,)
        Predicted default probabilities.
    threshold : float
        Decline at or above this.
    cost_matrix : array-like of shape (2, 2)
        Indexed ``[true][predicted]``.

    Returns
    -------
    dict
        Counts, rates and costs. ``cost_per_approved`` and
        ``bad_rate_among_approved`` are NaN when nothing is approved — a book
        with no loans has no cost per loan, and reporting zero would read as
        a free lunch.

    Examples
    --------
    >>> result = decision_costs([0, 0, 1, 0], [0.05, 0.10, 0.80, 0.70], 0.5, cost_matrix(5.0))
    >>> result["n_approved"], result["total_cost"], result["cost_per_approved"]
    (2, 1.0, 0.5)
    """
    truth = np.asarray(y_true).astype(int)
    probability = np.asarray(y_prob, dtype=float)
    costs = np.asarray(cost_matrix, dtype=float)

    approved = probability < threshold
    approved_bad = int(np.sum(approved & (truth == 1)))
    approved_good = int(np.sum(approved & (truth == 0)))
    declined_bad = int(np.sum(~approved & (truth == 1)))
    declined_good = int(np.sum(~approved & (truth == 0)))

    total = (
        approved_good * costs[0, 0]
        + declined_good * costs[0, 1]
        + approved_bad * costs[1, 0]
        + declined_bad * costs[1, 1]
    )
    n_approved = approved_good + approved_bad
    return {
        "threshold": float(threshold),
        "n": int(truth.size),
        "n_approved": n_approved,
        "approval_rate": n_approved / truth.size if truth.size else float("nan"),
        "n_approved_bad": approved_bad,
        "n_approved_good": approved_good,
        "n_declined_bad": declined_bad,
        "n_declined_good": declined_good,
        "bad_rate_among_approved": approved_bad / n_approved if n_approved else float("nan"),
        "total_cost": float(total),
        "cost_per_application": float(total / truth.size) if truth.size else float("nan"),
        "cost_per_approved": float(total / n_approved) if n_approved else float("nan"),
    }
