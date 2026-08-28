"""Adverse-action reason codes, and the registered gate on their stability.

Holds: the rule turning SHAP contributions into the reasons a declined applicant
is given, a perturbation that asks whether those reasons are stable, and the
threshold deciding whether they are usable at all.

Rung R5 differs from the rungs around it. R1, R3 and R4 change the model and the
metrics move. R5 changes nothing — it is a **gate**. A model that fails it is not
a slightly worse model, it is one whose declines cannot be explained to the person
receiving them, and an explanation that changes when the applicant's file is
nudged is not an explanation.

The threshold below is registered before any flip rate is measured, and it is not
revised afterwards. This is the one place in the project where that discipline is
load-bearing: a gate tuned until the model passes is not a gate, and the tuning
would be invisible in the final report.

The perturbation is deliberately large. Half a standard deviation is not
measurement noise, it is a materially different applicant, so passing at that
scale is a strong claim and failing is not embarrassing. The full flip rate curve
across smaller perturbations is reported alongside, because the registered point
is one reading from a curve and the curve is the more informative artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "FLIP_RATE_GATE",
    "GATE_REGISTRATION",
    "PERTURBATION_SIGMA",
    "TOP_K",
    "flip_rate",
    "gate_verdict",
    "perturb",
    "reason_frame",
    "top_reasons",
]

#: Reasons given per declined application. Three is the common regulatory
#: expectation and the number a person can act on.
TOP_K = 3

#: Perturbation scale, in standard deviations of each numeric feature.
PERTURBATION_SIGMA = 0.5

#: Maximum share of declined applicants whose top-:data:`TOP_K` reason *set* may
#: change under that perturbation. Inclusive: a flip rate exactly equal to this
#: passes.
FLIP_RATE_GATE = 0.10

GATE_REGISTRATION = (
    "Registered before measurement, in this module, and not revised afterwards. "
    f"A model passes rung R5 if at most {FLIP_RATE_GATE:.0%} of declined applicants see "
    f"their top-{TOP_K} adverse-action reason set change when every numeric feature in "
    f"their file is independently perturbed by a normal draw of {PERTURBATION_SIGMA} "
    "standard deviations. Order within the set does not count as a change: the applicant "
    "is told the same three things, and which is printed first is presentation. "
    "Substituting one reason for another does count, because the applicant is then told "
    "to fix something different. The verdict is reported whichever way it falls."
)


def top_reasons(
    contributions: ArrayLike, feature_names: list[str], k: int = TOP_K
) -> list[tuple[str, ...]]:
    """Turn per-feature SHAP contributions into adverse-action reasons.

    Only contributions that push risk *upwards* can be reasons. A feature that
    lowered an applicant's predicted default probability is not why they were
    declined, and listing it would be actively misleading — the applicant would
    be told to change something that was helping them.

    That is why a row can return fewer than ``k`` reasons, or none. An applicant
    declined despite every feature helping is a rare and interesting case, and it
    should surface rather than be padded out to three.

    Parameters
    ----------
    contributions : array-like of shape (n, n_features)
        SHAP values on the model's output scale, one row per application.
    feature_names : list of str
        Column names, in the matrix's column order.
    k : int
        Reasons per application.

    Returns
    -------
    list of tuple of str
        One tuple per row, most influential first, shortened where fewer than
        ``k`` contributions are positive.

    Raises
    ------
    ValueError
        If the name list does not match the matrix width.

    Examples
    --------
    >>> import numpy as np
    >>> top_reasons(np.array([[0.2, 0.5, -0.9]]), ["a", "b", "c"], k=2)
    [('b', 'a')]
    """
    matrix = np.asarray(contributions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"contributions has {matrix.shape[-1]} columns but {len(feature_names)} "
            f"feature names were given"
        )

    names = np.asarray(feature_names, dtype=object)
    # Stable sort so equal contributions resolve by column order rather than by
    # whatever the sort happened to do, which keeps a rerun's reasons identical.
    order = np.argsort(-matrix, axis=1, kind="stable")
    out: list[tuple[str, ...]] = []
    for row, ranking in zip(matrix, order, strict=True):
        top = [index for index in ranking[:k] if row[index] > 0]
        out.append(tuple(names[top]))
    return out


def reason_frame(
    identifiers: ArrayLike,
    contributions: ArrayLike,
    feature_names: list[str],
    k: int = TOP_K,
) -> pd.DataFrame:
    """Tabulate reasons one row per application, ready to persist.

    Parameters
    ----------
    identifiers : array-like of shape (n,)
        Application identifiers.
    contributions : array-like of shape (n, n_features)
        SHAP values.
    feature_names : list of str
        Column names in matrix order.
    k : int
        Reasons per application.

    Returns
    -------
    pandas.DataFrame
        ``SK_ID_CURR``, ``reason_1`` … ``reason_k``, and ``n_reasons``. Missing
        reasons are empty strings rather than nulls, so the table renders in a
        letter template without special-casing.

    Examples
    --------
    >>> import numpy as np
    >>> frame = reason_frame([1], np.array([[0.2, 0.5, -0.9]]), ["a", "b", "c"], k=3)
    >>> frame["reason_1"].iloc[0], int(frame["n_reasons"].iloc[0])
    ('b', 2)
    """
    chosen = top_reasons(contributions, feature_names, k=k)
    table = pd.DataFrame({"SK_ID_CURR": np.asarray(identifiers)})
    for position in range(k):
        table[f"reason_{position + 1}"] = [
            row[position] if position < len(row) else "" for row in chosen
        ]
    table["n_reasons"] = [len(row) for row in chosen]
    return table


def flip_rate(baseline: list[tuple[str, ...]], perturbed: list[tuple[str, ...]]) -> float:
    """Share of applications whose reason *set* changed.

    Compared as sets. Reordering is not a flip — the applicant is told the same
    things and the print order is presentation. Substitution is a flip, because
    the applicant is now told to fix something different.

    Parameters
    ----------
    baseline, perturbed : list of tuple of str
        Reasons before and after perturbation, aligned row for row.

    Returns
    -------
    float
        Between 0 and 1.

    Raises
    ------
    ValueError
        If the two lists differ in length.

    Examples
    --------
    Reordering is not a change; substitution is.

    >>> flip_rate([("a", "b", "c")], [("c", "a", "b")])
    0.0
    >>> flip_rate([("a", "b", "c")], [("a", "b", "z")])
    1.0
    """
    if len(baseline) != len(perturbed):
        raise ValueError(
            f"reason lists must be the same length; got {len(baseline)} and {len(perturbed)}"
        )
    if not baseline:
        return float("nan")
    changed = sum(
        set(before) != set(after) for before, after in zip(baseline, perturbed, strict=True)
    )
    return changed / len(baseline)


def gate_verdict(observed: float, threshold: float = FLIP_RATE_GATE) -> str:
    """Compare an observed flip rate against the registered threshold.

    A plain comparison, kept as a named function so the report never states a
    verdict the code did not compute.

    Parameters
    ----------
    observed : float
        Measured flip rate.
    threshold : float
        Registered maximum. Inclusive.

    Returns
    -------
    str
        ``"pass"`` or ``"fail"``.

    Examples
    --------
    >>> gate_verdict(0.04), gate_verdict(0.30)
    ('pass', 'fail')
    """
    return "pass" if observed <= threshold else "fail"


def perturb(
    frame: pd.DataFrame,
    features: list[str],
    sigma: float = PERTURBATION_SIGMA,
    seed: int = 20260827,
) -> pd.DataFrame:
    """Nudge every numeric feature by an independent normal draw.

    Each column is scaled by its own standard deviation, so a nudge means the
    same thing to an income in the hundreds of thousands and to a count in the
    single digits.

    Categorical columns are left untouched. There is no scale on which to move a
    category by half a standard deviation, and swapping one for another would be
    a different applicant rather than a perturbed one. Missing values stay
    missing: an absent field is not a field with an uncertain value, and filling
    it here would test imputation rather than explanation stability.

    Parameters
    ----------
    frame : pandas.DataFrame
        Applications to perturb.
    features : list of str
        Columns to consider.
    sigma : float
        Scale, in per-column standard deviations.
    seed : int
        Recorded with the resulting flip rate.

    Returns
    -------
    pandas.DataFrame
        A copy.

    Examples
    --------
    >>> import pandas as pd
    >>> moved = perturb(pd.DataFrame({"x": [1.0, 2.0], "g": ["a", "b"]}), ["x", "g"])
    >>> moved["g"].tolist()
    ['a', 'b']
    """
    rng = np.random.default_rng(seed)
    moved = frame.copy()
    for column in features:
        series = moved[column]
        if isinstance(series.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(
            series
        ):
            continue
        values = series.astype("float64")
        spread = float(values.std(skipna=True))
        if not np.isfinite(spread) or spread == 0.0:
            continue
        moved[column] = values + rng.normal(0.0, sigma * spread, len(values))
    return moved
