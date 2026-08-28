"""The stratified split, and the reason it is not a temporal one.

Holds: a three-way stratified assignment of applications to train, calibration
and test, and the seed that makes it reproducible.

``spine.splitting`` is not used here, and the omission is deliberate. Home
Credit's ``application_train.csv`` carries no application date — ``DAYS_BIRTH``,
``DAYS_EMPLOYED`` and the rest are all measured *relative* to the application, so
there is no absolute axis to order applications along. A rolling-origin split
needs an origin. Inventing one, from row order or from bureau recency, would
manufacture a temporal claim the data cannot support and would make every
out-of-time statement in the report false.

So this is a cross-section and it is split as one. What a temporal split exists
to prevent — information from after the decision reaching the model — is handled
instead by :mod:`adil.features`, where every satellite aggregate is restricted to
records knowable at application time and the restriction is tested.

Three ways rather than two, because the challenger needs a calibration set the
gradient booster has never seen. Calibrating on training predictions produces a
reliability curve that flatters the model, and a cost-based approval threshold
built on it would be wrong in a direction nobody notices.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ["PROPORTIONS", "SEED", "SPLIT_NAMES", "stratified_split"]

#: Recorded, not incidental. Every experiment in this project reports the seed it
#: ran under, and this is the one the split ran under.
SEED = 20260827

#: Train / calibration / test. The calibration share is large enough that an
#: isotonic fit on an ~8% base rate has enough positives to be stable.
PROPORTIONS: dict[str, float] = {"train": 0.60, "calibration": 0.20, "test": 0.20}

SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "test")


def stratified_split(
    frame: pd.DataFrame,
    target_column: str = "TARGET",
    id_column: str = "SK_ID_CURR",
    seed: int = SEED,
) -> pd.DataFrame:
    """Assign each application to train, calibration or test.

    Stratified on the target, because at an ~8% base rate an unstratified split
    can shift the rate between folds by enough to move a Brier score on its own.

    Parameters
    ----------
    frame : pandas.DataFrame
        Must carry ``id_column`` and ``target_column``.
    target_column : str
        Binary outcome to stratify on.
    id_column : str
        Application identifier, carried through so the assignment can be joined
        back and checked for overlap.
    seed : int
        Recorded in the metrics artifact alongside any result computed on this
        split.

    Returns
    -------
    pandas.DataFrame
        Columns ``id_column``, ``target_column`` and ``split``, in the input's
        row order.

    Raises
    ------
    KeyError
        If either required column is absent.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"SK_ID_CURR": range(100), "TARGET": [0, 1] * 50})
    >>> assignment = stratified_split(frame)
    >>> int(assignment["split"].value_counts()["train"])
    60
    """
    for column in (id_column, target_column):
        if column not in frame.columns:
            raise KeyError(f"{column!r} is not a column of the frame; found {list(frame.columns)}")

    labels = frame[[id_column, target_column]].copy()
    held_out_share = PROPORTIONS["calibration"] + PROPORTIONS["test"]

    train_index, held_out_index = train_test_split(
        labels.index,
        test_size=held_out_share,
        stratify=labels[target_column],
        random_state=seed,
    )
    calibration_index, test_index = train_test_split(
        held_out_index,
        test_size=PROPORTIONS["test"] / held_out_share,
        stratify=labels.loc[held_out_index, target_column],
        random_state=seed,
    )

    labels["split"] = pd.Series(index=labels.index, dtype="object")
    labels.loc[train_index, "split"] = "train"
    labels.loc[calibration_index, "split"] = "calibration"
    labels.loc[test_index, "split"] = "test"
    return labels
