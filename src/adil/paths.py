"""Where the raw tables are, and where this project's artifacts land.

Holds the one answer to "where is the data" — resolved through
:func:`spine.io.data_root`, so ADIL and its sibling repos cannot drift — plus the
project-scoped directories underneath it.

Processed artifacts live under ``<DATA_ROOT>/processed/adil/`` rather than inside
the repository. The shared store sits outside every git repo and is never
committed, and namespacing by project stops four repos writing over each other's
frames. Reports and metrics *are* committed, so they stay in the repository.
"""

from __future__ import annotations

from pathlib import Path

from spine.io import data_root

__all__ = [
    "HOME_CREDIT_TABLES",
    "german_credit_dir",
    "home_credit_dir",
    "metrics_dir",
    "processed_dir",
    "project_root",
    "reports_dir",
]

#: The seven Home Credit tables ADIL reads, and the column each is keyed on.
#: ``bureau_balance`` is keyed on ``SK_ID_BUREAU`` and reaches an application
#: only through ``bureau``, which is why it is the one table not joined directly.
HOME_CREDIT_TABLES: dict[str, str] = {
    "application_train": "SK_ID_CURR",
    "bureau": "SK_ID_CURR",
    "bureau_balance": "SK_ID_BUREAU",
    "previous_application": "SK_ID_CURR",
    "POS_CASH_balance": "SK_ID_CURR",
    "credit_card_balance": "SK_ID_CURR",
    "installments_payments": "SK_ID_CURR",
}


def project_root() -> Path:
    """Locate the repository root.

    Resolved from this file's position rather than the working directory, so a
    notebook running in ``notebooks/`` and a script running at the root agree.

    Returns
    -------
    pathlib.Path
        The ``02-adil`` directory.

    Examples
    --------
    >>> project_root().name
    '02-adil'
    """
    return Path(__file__).resolve().parents[2]


def home_credit_dir() -> Path:
    """Locate the raw Home Credit tables.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/raw/home-credit``.

    Raises
    ------
    RuntimeError
        If ``DATA_ROOT`` is not configured, or the directory is absent.
    """
    directory = data_root(project_root()) / "raw" / "home-credit"
    if not directory.is_dir():
        raise RuntimeError(
            f"expected the Home Credit tables at {directory}, which does not exist; "
            f"see data/README.md for the download checklist"
        )
    return directory


def german_credit_dir() -> Path:
    """Locate the raw UCI German Credit files.

    German Credit is used only in notebook 02, where its 1,000 rows make the
    weight-of-evidence arithmetic checkable by hand. It is not a modelling
    dataset here: different target, different features, different country.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/raw/german-credit``.

    Raises
    ------
    RuntimeError
        If ``DATA_ROOT`` is not configured, or the directory is absent.
    """
    directory = data_root(project_root()) / "raw" / "german-credit"
    if not directory.is_dir():
        raise RuntimeError(
            f"expected the German Credit files at {directory}, which does not exist; "
            f"see data/README.md for the download checklist"
        )
    return directory


def processed_dir() -> Path:
    """Locate the directory for this project's processed artifacts, creating it.

    Returns
    -------
    pathlib.Path
        ``<DATA_ROOT>/processed/adil``.
    """
    directory = data_root(project_root()) / "processed" / "adil"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def metrics_dir() -> Path:
    """Locate the committed directory holding every reported number, creating it.

    Returns
    -------
    pathlib.Path
        ``<repo>/metrics``.
    """
    directory = project_root() / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def reports_dir() -> Path:
    """Locate the committed markdown reports directory, creating it.

    Returns
    -------
    pathlib.Path
        ``<repo>/reports``.
    """
    directory = project_root() / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
