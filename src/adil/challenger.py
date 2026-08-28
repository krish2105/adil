"""The gradient boosting challenger, and the rungs of the constraint ladder.

Holds: the LightGBM configuration shared by every rung, the rung declarations
themselves, and the two utilities the notebooks need — a design matrix LightGBM
will accept, and gain importance for the feature cap.

The ladder exists to answer the project's question. R1 gives the challenger its
best shot: every candidate feature, no shape constraint. Each rung after it adds
exactly one regulatory constraint and the metrics move by exactly the amount that
constraint cost.

======  ===========================================================
Rung    What it adds
======  ===========================================================
R1      LightGBM, unconstrained, full candidate pool
R3      Monotone constraints on directionally-agreed features
R4      Feature count capped at the scorecard's
======  ===========================================================

R2 is missing from that table on purpose. Calibration is applied to every rung and
to the scorecard alike, because a miscalibrated model cannot support a cost-based
threshold at all — it is a precondition of notebook 06 rather than a constraint
the ladder imposes. R5 and R6 change no model: R5 is a pass/fail gate on reason
code stability, and R6 moves the cutoff, not the fit.

Number of boosting rounds is chosen by cross-validation *within the training
split* and the model is then refitted on all of it. Early stopping against a
held-out slice would have been simpler and would have handed the scorecard an
unearned advantage, since it saw all 184,506 training rows and the challenger
would have seen fewer. Both models see exactly the same rows. The calibration and
test splits are touched by neither.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from adil.scorecard import MAX_CHARACTERISTICS
from adil.split import SEED

__all__ = [
    "BASE_PARAMS",
    "RUNGS",
    "Rung",
    "best_rounds",
    "design_matrix",
    "fit",
    "gain_importance",
]

#: Shared across every rung, so a metric difference between rungs is the
#: constraint and not a hyperparameter. Deliberately unremarkable settings: this
#: project asks what constraints cost, not how high the AUC can be pushed, and a
#: tuned challenger would confound the two.
BASE_PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 10,
    # A fixed seed is not on its own enough. LightGBM builds histograms across
    # threads and picks its row-wise/col-wise strategy from a runtime heuristic, so
    # two runs on identical data can differ in the sixth decimal — small enough to
    # look like nothing and large enough to move every number in every report. These
    # two settings cost a little speed and buy a pipeline whose results are the same
    # on Tuesday as they were on Monday.
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass(frozen=True)
class Rung:
    """One rung of the constraint ladder.

    Parameters
    ----------
    name : str
        ``R1``, ``R3``, ``R4``.
    description : str
        What this rung adds to the one below it.
    monotone : bool
        Whether the declared credit directions in :mod:`adil.constraints` are
        imposed.
    feature_cap : int or None
        Maximum features, or ``None`` for the full pool. Where set, it is
        :data:`adil.scorecard.MAX_CHARACTERISTICS` — the same declared budget the
        scorecard is held to, so neither model is given more room than the other.
        The features kept are the highest-gain ones from the rung below, which
        makes the cap a selection exercise rather than an arbitrary truncation.
        Notebook 03 asserts that the scorecard actually used this many; if IV
        selection had stopped short of the cap, the comparison would be uneven
        and the assertion fails rather than quietly proceeding.

    Examples
    --------
    >>> RUNGS[0].name
    'R1'
    """

    name: str
    description: str
    monotone: bool
    feature_cap: int | None


RUNGS: tuple[Rung, ...] = (
    Rung(
        name="R1",
        description=(
            "LightGBM on the full candidate pool with no shape constraint. The "
            "challenger's best case, and the number every later rung is charged against."
        ),
        monotone=False,
        feature_cap=None,
    ),
    Rung(
        name="R3",
        description=(
            "Monotone constraints on the features whose credit direction was declared in "
            "adil.constraints before fitting. Buys a model whose response to a "
            "delinquency measure can be stated to a customer without qualification."
        ),
        monotone=True,
        feature_cap=None,
    ),
    Rung(
        name="R4",
        description=(
            "Monotone, and restricted to as many features as the scorecard uses, chosen "
            "by gain importance from R3. Puts the two models on an equal feature budget "
            "so the remaining gap is model class rather than data volume."
        ),
        monotone=True,
        feature_cap=MAX_CHARACTERISTICS,
    ),
)


def design_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Coerce a frame into something LightGBM will accept, in a fixed column order.

    Two conversions, both of which otherwise fail at fit time rather than here.
    Text columns become pandas categoricals, which LightGBM splits natively.
    Nullable extension dtypes — ``Int64`` and ``boolean``, which arrive from
    parquet — become ``float64``, so missing values reach LightGBM as NaN and are
    routed rather than rejected.

    Parameters
    ----------
    frame : pandas.DataFrame
        The modelling frame.
    features : list of str
        Columns to include, in the order the design matrix should use them.
        Order is load-bearing: LightGBM matches ``monotone_constraints`` to
        columns by position, so a reordering silently constrains the wrong one.

    Returns
    -------
    pandas.DataFrame
        A copy, converted.

    Raises
    ------
    KeyError
        If any requested feature is absent.

    Examples
    --------
    >>> import pandas as pd
    >>> design = design_matrix(pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}), ["b", "a"])
    >>> design.columns.tolist(), isinstance(design["a"].dtype, pd.CategoricalDtype)
    (['b', 'a'], True)
    """
    absent = [name for name in features if name not in frame.columns]
    if absent:
        raise KeyError(f"features absent from the frame: {absent}")

    design = frame[features].copy()
    for column in features:
        dtype = design[column].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            continue
        if not pd.api.types.is_numeric_dtype(dtype):
            design[column] = design[column].astype("category")
        elif str(dtype) in ("Int64", "boolean", "Float64"):
            design[column] = design[column].astype("float64")
    return design


def best_rounds(
    design: pd.DataFrame,
    y: ArrayLike,
    monotone: list[int] | None = None,
    max_rounds: int = 2000,
    folds: int = 5,
    patience: int = 100,
    params: dict[str, object] | None = None,
) -> int:
    """Choose the number of boosting rounds by cross-validation within the training split.

    Chosen this way rather than by early stopping against a held-out slice so
    that the challenger is refitted on every training row the scorecard saw. A
    comparison where one model trains on less data measures the split, not the
    model class.

    Parameters
    ----------
    design : pandas.DataFrame
        Training design matrix, from :func:`design_matrix`.
    y : array-like
        Training outcomes.
    monotone : list of int, optional
        Constraint per column, positionally matched.
    max_rounds : int
        Upper bound on rounds searched.
    folds : int
        Stratified folds.
    patience : int
        Rounds without improvement before stopping.
    params : dict, optional
        Overrides for :data:`BASE_PARAMS`.

    Returns
    -------
    int
        Rounds to refit with; at least 1.
    """
    settings = {**BASE_PARAMS, **(params or {})}
    if monotone is not None:
        settings["monotone_constraints"] = monotone
    dataset = lgb.Dataset(design, label=np.asarray(y), free_raw_data=False)
    history = lgb.cv(
        settings,
        dataset,
        num_boost_round=max_rounds,
        nfold=folds,
        stratified=True,
        seed=settings["seed"],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    lengths = [len(values) for key, values in history.items() if key.endswith("-mean")]
    return max(1, max(lengths))


def fit(
    frame: pd.DataFrame,
    y: ArrayLike,
    features: list[str],
    monotone: list[int] | None = None,
    num_boost_round: int = 400,
    params: dict[str, object] | None = None,
) -> lgb.Booster:
    """Fit one rung's model.

    Parameters
    ----------
    frame : pandas.DataFrame
        Training rows. Converted by :func:`design_matrix`.
    y : array-like
        Training outcomes; 1 is default.
    features : list of str
        Columns to use, in design-matrix order.
    monotone : list of int, optional
        One constraint per feature, in the same order. ``None`` leaves the model
        unconstrained.
    num_boost_round : int
        Rounds, normally from :func:`best_rounds`.
    params : dict, optional
        Overrides for :data:`BASE_PARAMS`.

    Returns
    -------
    lightgbm.Booster

    Raises
    ------
    ValueError
        If ``monotone`` is given and does not match ``features`` in length. A
        short or long vector would be applied positionally to the wrong columns.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> frame = pd.DataFrame({"x": rng.uniform(size=500)})
    >>> y = rng.binomial(1, frame["x"])
    >>> model = fit(frame, y, ["x"], monotone=[1], num_boost_round=10)
    >>> model.num_trees() > 0
    True
    """
    if monotone is not None and len(monotone) != len(features):
        raise ValueError(
            f"monotone has {len(monotone)} entries for {len(features)} features; "
            f"a positional mismatch would constrain the wrong columns"
        )
    settings = {**BASE_PARAMS, **(params or {})}
    if monotone is not None:
        settings["monotone_constraints"] = monotone
    design = design_matrix(frame, features)
    dataset = lgb.Dataset(design, label=np.asarray(y), free_raw_data=False)
    return lgb.train(settings, dataset, num_boost_round=num_boost_round)


def gain_importance(model: lgb.Booster, features: list[str]) -> pd.Series:
    """Total split gain per feature, descending.

    Gain rather than split count: a feature used once on a decisive split matters
    more than one used often on marginal ones, and the cap at rung R4 should keep
    the first.

    Parameters
    ----------
    model : lightgbm.Booster
        A fitted model.
    features : list of str
        The feature names in the order the model was fitted with.

    Returns
    -------
    pandas.Series
        Indexed by feature, sorted descending. Features the model never split on
        appear with zero rather than being dropped.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> frame = pd.DataFrame({"x": rng.uniform(size=500), "junk": rng.normal(size=500)})
    >>> y = rng.binomial(1, frame["x"])
    >>> model = fit(frame, y, ["x", "junk"], num_boost_round=30)
    >>> gain_importance(model, ["x", "junk"]).idxmax()
    'x'
    """
    gains = pd.Series(model.feature_importance(importance_type="gain"), index=features, dtype=float)
    return gains.sort_values(ascending=False)
