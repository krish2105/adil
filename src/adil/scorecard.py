r"""The weight-of-evidence scorecard, and the arithmetic that verifies it.

Holds: an independent implementation of weight of evidence and information
value, the rule deciding which columns may become characteristics, and the
configuration of optbinning's scorecard.

The WoE and IV functions here duplicate what optbinning already computes, on
purpose. A library checked against itself is not checked. These are written from
the formula, pinned to hand-computed values in ``tests/test_scorecard.py``, and
notebook 02 compares them against optbinning's own binning table on a real
feature — the same discipline SPINE applies in ``reports/proof.json``, where the
independent source is never SPINE.

The scorecard is the baseline the gradient booster must beat, and a weak baseline
is a form of dishonesty. It gets monotonic optimal binning, information-value
selection and proper points scaling rather than a logistic regression thrown at
raw columns.

Notation follows the credit convention throughout: the *event* is default
(``TARGET = 1``), a *nonevent* is a good account, and

.. math:: \mathrm{WoE}_i = \ln \frac{\%\,\text{nonevent}_i}{\%\,\text{event}_i}

so a bin richer in good accounts scores positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "BASE_ODDS",
    "BASE_POINTS",
    "IDENTIFIERS",
    "IV_FLOOR",
    "MAX_CHARACTERISTICS",
    "MAX_CORRELATION",
    "PDO",
    "PROTECTED_ATTRIBUTES",
    "TARGET",
    "bin_statistics",
    "candidate_features",
    "information_value",
    "pdo_scaling",
    "select_characteristics",
    "weight_of_evidence",
]

#: Sex and age. Both are genuinely protected under essentially every consumer
#: credit regime — they are not proxies, whatever the project brief assumed — and
#: neither may enter a model as an input, because deciding on them is disparate
#: treatment. They are held aside for measurement only, which is not the same as
#: making the model fair: proxies for both survive among the remaining columns,
#: and notebook 05 reports the disparity that persists.
PROTECTED_ATTRIBUTES: tuple[str, ...] = ("CODE_GENDER", "DAYS_BIRTH")

IDENTIFIERS: tuple[str, ...] = ("SK_ID_CURR",)

TARGET = "TARGET"

#: Conventional floor for a characteristic worth keeping. Below roughly 0.02 a
#: feature is not distinguishing anything a scorecard can act on.
IV_FLOOR = 0.02

#: A scorecard a credit officer can read is a scorecard of ten to twenty
#: characteristics, not five hundred. This cap is also what notebook 03 imposes
#: on the challenger at rung R4, so the two are compared at equal feature budget.
MAX_CHARACTERISTICS = 20

#: Two characteristics correlating above this on the weight-of-evidence scale are
#: telling the scorecard the same thing twice. The lower-IV one is dropped, which
#: keeps the logistic coefficients interpretable — a points table where two rows
#: cancel each other out cannot support an adverse-action reason.
MAX_CORRELATION = 0.7

#: Points to double the odds, the base odds, and the score those odds sit at.
#: The industry-standard 20/50:1/600 triple. Nothing downstream depends on the
#: choice — scaling is affine and monotone, so it moves scores, never rankings,
#: probabilities or decisions.
PDO = 20
BASE_ODDS = 50
BASE_POINTS = 600


def weight_of_evidence(
    n_event: float, n_nonevent: float, total_event: float, total_nonevent: float
) -> float:
    r"""Weight of evidence for one bin.

    .. math:: \ln \frac{n_{\text{nonevent}} / N_{\text{nonevent}}}
                        {n_{\text{event}} / N_{\text{event}}}

    Parameters
    ----------
    n_event, n_nonevent : float
        Counts in this bin.
    total_event, total_nonevent : float
        Counts across the whole sample.

    Returns
    -------
    float
        The weight of evidence, or NaN where a cell is empty. NaN rather than a
        smoothed value: an empty cell means this sample cannot answer the
        question for this bin, whereas a smoothed number would assert that the
        bin is very safe or very risky, and the two call for different responses.

    Examples
    --------
    A bin holding a fifth of the defaults but a third of the good accounts.

    >>> round(weight_of_evidence(20, 80, 30, 270), 6)
    -0.81093
    """
    if n_event <= 0 or n_nonevent <= 0 or total_event <= 0 or total_nonevent <= 0:
        return float("nan")
    return float(np.log((n_nonevent / total_nonevent) / (n_event / total_event)))


def bin_statistics(bins: pd.DataFrame) -> pd.DataFrame:
    """Attach counts, rates, weight of evidence and IV contribution to a bin table.

    Parameters
    ----------
    bins : pandas.DataFrame
        Must carry ``n_event`` and ``n_nonevent``. Any other columns, such as a
        bin label, are carried through untouched.

    Returns
    -------
    pandas.DataFrame
        The input plus ``n``, ``event_rate``, ``share_event``, ``share_nonevent``,
        ``woe`` and ``iv_contribution``.

    Raises
    ------
    KeyError
        If either count column is absent.

    Examples
    --------
    >>> import pandas as pd
    >>> table = bin_statistics(
    ...     pd.DataFrame({"bin": ["A", "B"], "n_event": [20, 10], "n_nonevent": [80, 190]})
    ... )
    >>> float(round(table["woe"].iloc[0], 6)), float(round(table["iv_contribution"].iloc[0], 6))
    (-0.81093, 0.300345)
    """
    for column in ("n_event", "n_nonevent"):
        if column not in bins.columns:
            raise KeyError(
                f"{column!r} is not a column of the bin table; found {list(bins.columns)}"
            )

    table = bins.copy()
    total_event = float(table["n_event"].sum())
    total_nonevent = float(table["n_nonevent"].sum())

    table["n"] = table["n_event"] + table["n_nonevent"]
    table["event_rate"] = table["n_event"] / table["n"]
    table["share_event"] = table["n_event"] / total_event
    table["share_nonevent"] = table["n_nonevent"] / total_nonevent
    table["woe"] = [
        weight_of_evidence(event, nonevent, total_event, total_nonevent)
        for event, nonevent in zip(table["n_event"], table["n_nonevent"], strict=True)
    ]
    table["iv_contribution"] = (table["share_nonevent"] - table["share_event"]) * table["woe"]
    return table


def information_value(bins: pd.DataFrame) -> float:
    r"""Information value of a binned characteristic.

    .. math:: \mathrm{IV} = \sum_i
              (\%\,\text{nonevent}_i - \%\,\text{event}_i)\ \mathrm{WoE}_i

    Bins whose weight of evidence is undefined contribute nothing rather than
    propagating NaN through the total, so one empty cell does not erase an
    otherwise informative characteristic. A characteristic with such a bin is
    still worth a second look, which is why :func:`bin_statistics` keeps the NaN
    visible at bin level.

    Parameters
    ----------
    bins : pandas.DataFrame
        As accepted by :func:`bin_statistics`.

    Returns
    -------
    float
        The information value, which is non-negative by construction.

    Examples
    --------
    >>> import pandas as pd
    >>> round(information_value(
    ...     pd.DataFrame({"n_event": [20, 10], "n_nonevent": [80, 190]})
    ... ), 6)
    0.577091
    """
    return float(bin_statistics(bins)["iv_contribution"].sum(skipna=True))


def candidate_features(frame: pd.DataFrame) -> list[str]:
    """List the columns permitted to become scorecard characteristics.

    Three exclusions, each for a different reason. Identifiers and the target are
    not features. The protected attributes are excluded because deciding on sex
    or age is disparate treatment, and they are kept in the frame only so
    notebook 05 can measure against them. Constant and all-null columns are
    excluded because they carry no evidence and would produce a single bin with
    an undefined weight of evidence.

    Parameters
    ----------
    frame : pandas.DataFrame
        The modelling frame.

    Returns
    -------
    list of str
        In the frame's column order, so the candidate set is stable across runs.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "CODE_GENDER": ["M", "F"],
    ...      "AMT_CREDIT": [10, 20]}
    ... )
    >>> candidate_features(frame)
    ['AMT_CREDIT']
    """
    excluded = {*IDENTIFIERS, TARGET, *PROTECTED_ATTRIBUTES}
    return [
        column
        for column in frame.columns
        if column not in excluded and frame[column].nunique(dropna=True) > 1
    ]


def pdo_scaling(
    pdo: float = PDO, base_odds: float = BASE_ODDS, base_points: float = BASE_POINTS
) -> tuple[float, float]:
    r"""Factor and offset for points-to-double-the-odds scaling.

    .. math:: \text{factor} = \frac{\text{PDO}}{\ln 2}, \qquad
              \text{offset} = \text{base points} - \text{factor} \ln(\text{base odds})

    so that ``score = offset + factor * ln(odds)``. The transformation is affine
    in the log-odds and strictly increasing, which is the whole point: it changes
    what a score looks like on a letter and changes nothing about ranking,
    probability or decision.

    Parameters
    ----------
    pdo : float
        Points that double the odds of being good.
    base_odds : float
        Odds anchoring the scale.
    base_points : float
        Score those odds sit at.

    Returns
    -------
    factor, offset : float

    Examples
    --------
    Twenty points to double the odds, anchored at 600 for odds of 50:1.

    >>> factor, offset = pdo_scaling()
    >>> round(offset + factor * float(np.log(50)), 6)
    600.0
    >>> round(offset + factor * float(np.log(100)) - 600, 6)
    20.0
    """
    factor = pdo / np.log(2)
    offset = base_points - factor * np.log(base_odds)
    return float(factor), float(offset)


def select_characteristics(
    information_values: pd.Series,
    correlation: pd.DataFrame,
    iv_floor: float = IV_FLOOR,
    max_characteristics: int = MAX_CHARACTERISTICS,
    max_correlation: float = MAX_CORRELATION,
) -> pd.DataFrame:
    """Choose the scorecard's characteristics, and record why each was kept or dropped.

    Greedy by information value: take the strongest remaining characteristic,
    then discard anything correlating with it above ``max_correlation``. Home
    Credit's bureau aggregates make this necessary rather than decorative —
    ``BUR_DAYS_CREDIT`` contributes a mean, a minimum and a maximum that rank
    consecutively on IV and say almost the same thing.

    The returned log is the artifact, not the list. "Why is this characteristic
    not in the scorecard?" is a question with an answer per row.

    Parameters
    ----------
    information_values : pandas.Series
        Information value indexed by feature name.
    correlation : pandas.DataFrame
        Square, symmetric correlation matrix over the same features. Correlate
        the weight-of-evidence transformed columns rather than the raw ones —
        that is the scale the logistic regression actually sees.
    iv_floor : float
        Below this a characteristic distinguishes too little to keep.
    max_characteristics : int
        Cap on the final count.
    max_correlation : float
        Absolute correlation above which the lower-IV characteristic is dropped.

    Returns
    -------
    pandas.DataFrame
        One row per candidate, sorted by information value descending. Columns:
        ``feature``, ``iv``, ``selected``, ``reason``.

    Examples
    --------
    ``b`` is a near-copy of ``a`` and loses to it; ``c`` is too weak to keep.

    >>> import pandas as pd
    >>> ivs = pd.Series({"a": 0.30, "b": 0.25, "c": 0.01})
    >>> corr = pd.DataFrame(
    ...     [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ...     index=["a", "b", "c"], columns=["a", "b", "c"],
    ... )
    >>> for row in select_characteristics(ivs, corr).itertuples():
    ...     print(f"{row.feature}  {row.selected}  {row.reason}")
    a  True  selected
    b  False  correlates 0.90 with 'a', which has the higher IV
    c  False  IV 0.010 below the floor of 0.020
    """
    ordered = information_values.sort_values(ascending=False)
    selected: list[str] = []
    rows = []

    for feature in ordered.index:
        iv = float(ordered[feature])
        if iv < iv_floor:
            reason = f"IV {iv:.3f} below the floor of {iv_floor:.3f}"
        elif len(selected) >= max_characteristics:
            reason = f"cap of {max_characteristics} characteristics already reached"
        else:
            rival = next(
                (
                    other
                    for other in selected
                    if abs(float(correlation.loc[feature, other])) > max_correlation
                ),
                None,
            )
            if rival is None:
                selected.append(feature)
                reason = "selected"
            else:
                shared = abs(float(correlation.loc[feature, rival]))
                reason = f"correlates {shared:.2f} with {rival!r}, which has the higher IV"
        rows.append(
            {"feature": feature, "iv": iv, "selected": reason == "selected", "reason": reason}
        )

    return pd.DataFrame(rows)
