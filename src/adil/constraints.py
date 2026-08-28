"""Monotone direction declarations for the challenger, fixed before it is fitted.

Holds: the credit directions ADIL is willing to assert in advance, and the
mapping from a feature list to LightGBM's ``monotone_constraints`` vector.

Rung R3 of the constraint ladder asks what monotonicity costs. The answer is only
worth anything if the directions were decided before the cost was known, so this
module is committed before notebook 03 runs and is not revised afterwards. The
temptation it exists to resist is obvious: fit, look at which constraints hurt,
quietly drop those, and report a smaller cost.

The rule for inclusion is deliberately narrow. A direction goes in this table only
if a credit officer would state it without looking at the data — a delinquency
measure increases risk, an external credit score decreases it. Directions that are
merely *empirically* true in Home Credit are left out, because a constraint
justified by the training data is not a constraint, it is a fit. Everything
undeclared gets zero, and most of the 567 features are undeclared.

Sign convention follows LightGBM: the model predicts P(default), so ``+1`` means
the predicted probability may not decrease as the feature increases, and ``-1``
means it may not increase.

One trap worth naming. Home Credit's ``DAYS_*`` columns count *backwards* from the
application and are negative, so a larger value is a more recent event, not an
older one. ``DAYS_EMPLOYED`` closer to zero is a shorter tenure, which is why its
direction is ``+1`` and not the ``-1`` that "longer employment is safer" suggests
at a glance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

__all__ = ["DIRECTIONS", "Direction", "constraint_table", "monotone_constraints"]


@dataclass(frozen=True)
class Direction:
    """One asserted credit direction.

    Parameters
    ----------
    pattern : str
        Regular expression matched against the feature name with
        :func:`re.search`. Anchored where the feature is a single column, left
        open where it must catch every aggregation of one source column.
    sign : int
        ``+1`` if predicted default risk may not decrease as the feature rises,
        ``-1`` if it may not increase.
    rationale : str
        Why a credit officer would assert this without seeing the data. Rendered
        into the model card, so it is the sentence that has to survive a viva.

    Examples
    --------
    >>> DIRECTIONS[0].sign
    -1
    """

    pattern: str
    sign: int
    rationale: str


#: Every direction ADIL is willing to assert in advance. Deliberately short.
DIRECTIONS: tuple[Direction, ...] = (
    Direction(
        pattern=r"^EXT_SOURCE_\d$",
        sign=-1,
        rationale=(
            "A normalised external credit score, where higher already means more "
            "creditworthy by the provider's own construction. Allowing default risk to "
            "rise with it would contradict the definition of the input, not merely the "
            "data."
        ),
    ),
    Direction(
        pattern=r"^AMT_INCOME_TOTAL$",
        sign=-1,
        rationale=(
            "Declared income. Holding the loan fixed, greater income is greater capacity "
            "to service it. This is the affordability premise the whole product rests on; "
            "a model that priced income the other way could not be defended to a customer."
        ),
    ),
    Direction(
        pattern=r"CREDIT_DAY_OVERDUE",
        sign=1,
        rationale=(
            "Days a bureau-reported credit is currently overdue. Arrears are the most "
            "direct observable evidence of distress, and their direction is not in "
            "dispute in any credit regime."
        ),
    ),
    Direction(
        pattern=r"AMT_CREDIT_MAX_OVERDUE|AMT_CREDIT_SUM_OVERDUE",
        sign=1,
        rationale=(
            "Amount overdue on bureau credits, current and historic maximum. The "
            "monetary counterpart of arrears; larger unpaid balances cannot indicate "
            "lower risk."
        ),
    ),
    Direction(
        pattern=r"CNT_CREDIT_PROLONG",
        sign=1,
        rationale=(
            "Times a bureau credit was prolonged. Prolongation is forbearance granted to "
            "a borrower who could not meet the original schedule, so it marks distress "
            "that has already been acted on by another lender."
        ),
    ),
    Direction(
        pattern=r"SK_DPD",
        sign=1,
        rationale=(
            "Days past due on the applicant's own prior Home Credit instalments and card "
            "balances, including the tolerance-adjusted variant. Missed payments to this "
            "lender cannot lower the probability of missing the next one."
        ),
    ),
    Direction(
        pattern=r"^BB_STATUS_[1-5]_SHARE$",
        sign=1,
        rationale=(
            "Share of bureau balance months in arrears buckets 1 to 5, where the bucket "
            "number counts months of delinquency and 5 is a write-off. Status 0 is a "
            "month paid on time and is left unconstrained: a high share of on-time months "
            "can simply mean a long open loan, so its direction is not agreed in advance."
        ),
    ),
    Direction(
        pattern=r"NAME_CONTRACT_STATUS_REFUSED_SHARE",
        sign=1,
        rationale=(
            "Share of the applicant's prior applications that were refused. A previous "
            "underwriter, seeing that file, declined it; that judgement is evidence and "
            "its direction is settled. Notebook 07 returns to this population, which is "
            "the only genuine record of refusal anywhere in the data."
        ),
    ),
    Direction(
        pattern=r"^DAYS_EMPLOYED$",
        sign=1,
        rationale=(
            "Days since employment began, counted backwards from the application and "
            "therefore negative. A value nearer zero is a shorter tenure, so risk rises "
            "with the column even though it falls with tenure. The sign is +1 for the "
            "same reason the column is confusing: the axis runs the other way."
        ),
    ),
)


def constraint_table(
    features: list[str], directions: tuple[Direction, ...] = DIRECTIONS
) -> pd.DataFrame:
    """Assign a direction to each feature and record which rule did it.

    The table, not the vector, is the artifact. "Why is this feature constrained,
    and who decided?" is a model-validation question, and it should have an
    answer per row.

    Parameters
    ----------
    features : list of str
        Feature names, in the order the model's design matrix uses them.
    directions : tuple of Direction
        Declarations to apply. Overridable so the rules can be tested.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``constraint``, ``matched_pattern``, ``rationale``,
        one row per input feature in input order.

    Raises
    ------
    ValueError
        If a feature matches more than one declaration. Two rules matching one
        feature means the declaration contradicts itself, and silently taking the
        first would conceal that.

    Examples
    --------
    >>> constraint_table(["EXT_SOURCE_1", "UNDECLARED"])["constraint"].tolist()
    [-1, 0]
    """
    rows = []
    for feature in features:
        matched = [d for d in directions if re.search(d.pattern, feature)]
        if len(matched) > 1:
            patterns = [d.pattern for d in matched]
            raise ValueError(
                f"{feature!r} matches more than one declared direction ({patterns}); "
                f"the declaration contradicts itself"
            )
        if matched:
            rows.append(
                {
                    "feature": feature,
                    "constraint": matched[0].sign,
                    "matched_pattern": matched[0].pattern,
                    "rationale": matched[0].rationale,
                }
            )
        else:
            rows.append(
                {
                    "feature": feature,
                    "constraint": 0,
                    "matched_pattern": "",
                    "rationale": "no agreed credit direction",
                }
            )
    return pd.DataFrame(rows)


def monotone_constraints(
    features: list[str], directions: tuple[Direction, ...] = DIRECTIONS
) -> list[int]:
    """Build LightGBM's ``monotone_constraints`` vector for a feature list.

    Parameters
    ----------
    features : list of str
        Feature names in design-matrix order. Order matters: LightGBM matches the
        vector to columns by position, so a mismatch silently constrains the
        wrong column.
    directions : tuple of Direction
        Declarations to apply.

    Returns
    -------
    list of int
        One entry per feature, in the same order. Zero where no direction is
        declared, which is the majority.

    Examples
    --------
    >>> monotone_constraints(["EXT_SOURCE_2", "AMT_CREDIT", "POS_SK_DPD_MAX"])
    [-1, 0, 1]
    """
    return constraint_table(features, directions)["constraint"].tolist()
