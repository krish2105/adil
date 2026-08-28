"""As-of-application feature construction over the seven Home Credit tables.

Holds: the declared as-of rule for each satellite table, the SQL that enforces
it, and the manifest that lets every feature name its own provenance.

Home Credit is a cross-section. ``application_train.csv`` carries no application
date — every ``DAYS_*`` column is measured *relative* to the application — so
there is no time axis to split on and :mod:`spine.splitting` is deliberately
unused by this project. Manufacturing a pseudo-date would fabricate a temporal
claim the data cannot support.

What replaces the temporal split is stricter about the thing a temporal split
exists to prevent. Each satellite table is filtered to records knowable when the
application was decided, the filter is an explicit ``WHERE`` clause rather than a
convention, and the number of rows it removes is recorded — *including when it
removes none*. A filter that turns out to be a no-op is a reported fact, not a
skipped step.

Aggregation runs in DuckDB against the CSVs directly. The seven tables are 5.3 GB
and 58 million satellite rows; only the app-level result, one row per
``SK_ID_CURR``, is ever loaded into pandas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from adil.paths import home_credit_dir

__all__ = [
    "CATEGORY_LEVEL_CAP",
    "NUMERIC_AGGREGATIONS",
    "SATELLITES",
    "FeatureSpec",
    "SatelliteTable",
    "build_frame",
    "build_select",
    "categorical_aggregations",
    "csv_source",
    "filter_report",
    "introspect",
    "manifest_frame",
    "numeric_aggregations",
]

#: Applied to every numeric column of every satellite table. Deliberately short:
#: four summaries the credit literature actually reads, rather than a wide net
#: of quantiles that would inflate the feature count without adding a reason
#: code anyone could act on.
NUMERIC_AGGREGATIONS: tuple[str, ...] = ("MIN", "MAX", "MEAN", "SUM")

#: Home Credit encodes "this never happened" as a positive day count of 365243 —
#: roughly a thousand years in the future — in several relative-day columns. It is a
#: sentinel, not a duration, and averaging it produces a number with no meaning. Every
#: aggregate over an affected column is computed on ``NULLIF(column, SENTINEL)``, and the
#: share of records carrying it is kept as a feature in its own right, because "no first
#: drawing ever occurred" is signal even though 365243 is not a quantity.
SENTINEL = 365243

#: Columns of ``application_train`` carrying :data:`SENTINEL`. Each is nulled and
#: paired with a ``_IS_SENTINEL`` flag: for ``DAYS_EMPLOYED`` the sentinel marks
#: pensioners and the unemployed, which is a credit-relevant state, so nulling it
#: without keeping the flag would discard 18% of the column's information.
APPLICATION_SENTINEL_COLUMNS: tuple[str, ...] = ("DAYS_EMPLOYED",)

#: A categorical column with more levels than this is not expanded into shares.
#: The cap exists because a share feature over a 100-level column is mostly
#: zeroes and cannot support an adverse-action reason. Skips are recorded.
CATEGORY_LEVEL_CAP = 30


@dataclass(frozen=True)
class SatelliteTable:
    """One Home Credit satellite table and the as-of rule that governs it.

    Parameters
    ----------
    name : str
        The table's file stem, without ``.csv``.
    prefix : str
        Short tag opening every feature name derived from this table, so a
        column's origin is legible without consulting the manifest.
    as_of_filter : str
        SQL boolean expression restricting the table to records knowable at
        application time. Never empty: a satellite table with nothing to filter
        would not be a satellite table.
    as_of_rationale : str
        Why this filter is the correct one. Reproduced in the data quality
        report, so the choice is defended in the artifact rather than in
        somebody's memory.
    key_column : str
        Column the aggregation groups to. Always ``SK_ID_CURR``: a feature that
        cannot reach an application cannot enter the frame.
    drop_columns : tuple of str
        Technical identifiers that are present but are not signal.
    sentinel_columns : tuple of str
        Columns using :data:`SENTINEL` for "this never happened". Aggregated over
        ``NULLIF``, with the sentinel's share retained as a separate feature.
    join_sql : str or None
        Replaces the plain CSV source when the table cannot reach an application
        directly. Only ``bureau_balance`` needs this — it is keyed on
        ``SK_ID_BUREAU`` and reaches an application through ``bureau``.

    Examples
    --------
    >>> bureau = next(t for t in SATELLITES if t.name == "bureau")
    >>> bureau.prefix
    'BUR'
    """

    name: str
    prefix: str
    as_of_filter: str
    as_of_rationale: str
    key_column: str = "SK_ID_CURR"
    drop_columns: tuple[str, ...] = field(default_factory=tuple)
    sentinel_columns: tuple[str, ...] = field(default_factory=tuple)
    join_sql: str | None = None


@dataclass(frozen=True)
class FeatureSpec:
    """One generated feature and everything needed to audit it.

    Parameters
    ----------
    feature : str
        Output column name.
    source_table, source_column, aggregation : str
        Provenance. ``source_column`` is ``"*"`` for a row count.
    expression : str
        The SQL that computes it.
    as_of_filter : str
        The filter in force when it was computed.

    Examples
    --------
    >>> bureau = next(t for t in SATELLITES if t.name == "bureau")
    >>> spec = numeric_aggregations(bureau, ["AMT_CREDIT_SUM"])[0]
    >>> spec.feature, spec.aggregation
    ('BUR_AMT_CREDIT_SUM_MIN', 'MIN')
    """

    feature: str
    source_table: str
    source_column: str
    aggregation: str
    expression: str
    as_of_filter: str


#: The six satellite tables, each with the rule that makes its aggregates
#: as-of-application. ``application_train`` is the spine of the frame and carries
#: no filter of its own — it *is* the application.
SATELLITES: tuple[SatelliteTable, ...] = (
    SatelliteTable(
        name="bureau",
        prefix="BUR",
        as_of_filter='"DAYS_CREDIT" < 0',
        as_of_rationale=(
            "DAYS_CREDIT is when the bureau-reported credit was opened, relative to this "
            "application. Only credits opened strictly before it were knowable. "
            "DAYS_CREDIT_ENDDATE is deliberately not filtered: a scheduled end date in the "
            "future is known at application time and is legitimate signal, whereas "
            "DAYS_ENDDATE_FACT is an actual closure and is dropped as a column below."
        ),
        drop_columns=("SK_ID_BUREAU", "DAYS_ENDDATE_FACT"),
    ),
    SatelliteTable(
        name="bureau_balance",
        prefix="BB",
        as_of_filter='"MONTHS_BALANCE" < 0',
        as_of_rationale=(
            "MONTHS_BALANCE counts months back from this application, so negative months "
            "are history and zero is the application month itself. Reached through bureau, "
            "which is filtered on the same principle before the join."
        ),
        drop_columns=("SK_ID_BUREAU",),
        join_sql=(
            'SELECT b."SK_ID_CURR", bb.* '
            "FROM {bureau_balance} bb "
            'JOIN (SELECT * FROM {bureau} WHERE "DAYS_CREDIT" < 0) b '
            'ON bb."SK_ID_BUREAU" = b."SK_ID_BUREAU"'
        ),
    ),
    SatelliteTable(
        name="previous_application",
        prefix="PRV",
        as_of_filter='"DAYS_DECISION" < 0',
        as_of_rationale=(
            "DAYS_DECISION is when the prior application was decided. A decision taken on "
            "or after the current application could not have informed it."
        ),
        drop_columns=("SK_ID_PREV",),
        sentinel_columns=(
            "DAYS_FIRST_DRAWING",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION",
        ),
    ),
    SatelliteTable(
        name="POS_CASH_balance",
        prefix="POS",
        as_of_filter='"MONTHS_BALANCE" < 0',
        as_of_rationale=(
            "MONTHS_BALANCE counts months back from this application. Zero is the "
            "application month, whose balance is not yet observable when deciding."
        ),
        drop_columns=("SK_ID_PREV",),
    ),
    SatelliteTable(
        name="credit_card_balance",
        prefix="CCB",
        as_of_filter='"MONTHS_BALANCE" < 0',
        as_of_rationale=(
            "MONTHS_BALANCE counts months back from this application. Zero is the "
            "application month, whose balance is not yet observable when deciding."
        ),
        drop_columns=("SK_ID_PREV",),
    ),
    SatelliteTable(
        name="installments_payments",
        prefix="INS",
        as_of_filter='"DAYS_INSTALMENT" < 0 AND "DAYS_ENTRY_PAYMENT" < 0',
        as_of_rationale=(
            "Both columns must be historic. An instalment scheduled before the application "
            "but paid after it reveals repayment behaviour that had not happened yet, so "
            "filtering on the due date alone would leak the outcome of the decision."
        ),
        drop_columns=("SK_ID_PREV",),
    ),
)


def csv_source(table_name: str) -> str:
    """Build the DuckDB reader expression for one raw table.

    Parameters
    ----------
    table_name : str
        File stem, without ``.csv``.

    Returns
    -------
    str
        A ``read_csv_auto`` call usable as a ``FROM`` clause.

    Examples
    --------
    >>> csv_source("bureau").startswith("read_csv_auto(")
    True
    """
    path = home_credit_dir() / f"{table_name}.csv"
    return f"read_csv_auto('{path}')"


def _slug(value: object) -> str:
    """Reduce a category level to a name that is safe in a column identifier.

    Parameters
    ----------
    value : object
        The level.

    Returns
    -------
    str
        Uppercase alphanumerics and underscores.

    Examples
    --------
    >>> _slug("Consumer credit / other")
    'CONSUMER_CREDIT_OTHER'
    >>> _slug(None)
    'MISSING'
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "MISSING"
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(value))).strip("_").upper()


def numeric_aggregations(table: SatelliteTable, numeric_columns: list[str]) -> list[FeatureSpec]:
    """Generate one spec per numeric column per aggregation, plus a row count.

    The key column and the table's declared ``drop_columns`` are excluded: an
    identifier aggregated into a mean is noise wearing a feature's clothes.

    Parameters
    ----------
    table : SatelliteTable
        The table being aggregated.
    numeric_columns : list of str
        Its numeric columns, as reported by :func:`introspect`.

    Returns
    -------
    list of FeatureSpec
        In stable order, so a regenerated frame has stable column order.

    Examples
    --------
    >>> bureau = next(t for t in SATELLITES if t.name == "bureau")
    >>> specs = numeric_aggregations(bureau, ["SK_ID_CURR", "AMT_ANNUITY"])
    >>> [s.feature for s in specs][:2]
    ['BUR_AMT_ANNUITY_MIN', 'BUR_AMT_ANNUITY_MAX']
    """
    excluded = {table.key_column, *table.drop_columns}
    kept = [column for column in numeric_columns if column not in excluded]

    def _operand(column: str) -> str:
        if column in table.sentinel_columns:
            return f'NULLIF("{column}", {SENTINEL})'
        return f'"{column}"'

    specs = [
        FeatureSpec(
            feature=f"{table.prefix}_{column}_{aggregation}",
            source_table=table.name,
            source_column=column,
            aggregation=(
                f"{aggregation} over NULLIF(column, {SENTINEL})"
                if column in table.sentinel_columns
                else aggregation
            ),
            expression=f"{aggregation}({_operand(column)})",
            as_of_filter=table.as_of_filter,
        )
        for column in kept
        for aggregation in NUMERIC_AGGREGATIONS
    ]
    specs += [
        FeatureSpec(
            feature=f"{table.prefix}_{column}_SENTINEL_SHARE",
            source_table=table.name,
            source_column=column,
            aggregation="SHARE",
            expression=f'AVG(CASE WHEN "{column}" = {SENTINEL} THEN 1.0 ELSE 0.0 END)',
            as_of_filter=table.as_of_filter,
        )
        for column in kept
        if column in table.sentinel_columns
    ]
    specs.append(
        FeatureSpec(
            feature=f"{table.prefix}_ROW_COUNT",
            source_table=table.name,
            source_column="*",
            aggregation="COUNT",
            expression="COUNT(*)",
            as_of_filter=table.as_of_filter,
        )
    )
    return specs


def categorical_aggregations(
    table: SatelliteTable, levels: dict[str, list[object]]
) -> list[FeatureSpec]:
    """Generate a share-of-records feature for each level of each categorical column.

    A share rather than a count, because the count is already carried by
    ``*_ROW_COUNT`` and a share is the quantity a credit analyst reads.

    Parameters
    ----------
    table : SatelliteTable
        The table being aggregated.
    levels : dict
        Column name to its observed levels, from :func:`introspect`.

    Returns
    -------
    list of FeatureSpec
        Names are made unique by suffixing a counter on the rare occasion two
        levels slug to the same identifier.

    Examples
    --------
    >>> bureau = next(t for t in SATELLITES if t.name == "bureau")
    >>> specs = categorical_aggregations(bureau, {"CREDIT_ACTIVE": ["Active"]})
    >>> specs[0].feature, specs[0].aggregation
    ('BUR_CREDIT_ACTIVE_ACTIVE_SHARE', 'SHARE')
    """
    specs: list[FeatureSpec] = []
    seen: set[str] = set()
    for column, values in levels.items():
        for value in values:
            name = f"{table.prefix}_{column}_{_slug(value)}_SHARE"
            if name in seen:
                name = f"{name}_{len(seen)}"
            seen.add(name)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                predicate = f'"{column}" IS NULL'
            else:
                predicate = f"\"{column}\" = '{str(value).replace(chr(39), chr(39) * 2)}'"
            specs.append(
                FeatureSpec(
                    feature=name,
                    source_table=table.name,
                    source_column=column,
                    aggregation="SHARE",
                    expression=f"AVG(CASE WHEN {predicate} THEN 1.0 ELSE 0.0 END)",
                    as_of_filter=table.as_of_filter,
                )
            )
    return specs


def build_select(table: SatelliteTable, specs: list[FeatureSpec], source: str | None = None) -> str:
    """Assemble the aggregation query for one table.

    The as-of filter appears as an explicit ``WHERE`` clause rather than being
    folded into the aggregate expressions, so an examiner reading the query sees
    the restriction in one place.

    Parameters
    ----------
    table : SatelliteTable
        The table being aggregated.
    specs : list of FeatureSpec
        What to compute.
    source : str, optional
        ``FROM`` expression. Defaults to the table's CSV reader, or its
        ``join_sql`` where one is declared.

    Returns
    -------
    str
        A query returning one row per ``key_column``.

    Examples
    --------
    >>> bureau = next(t for t in SATELLITES if t.name == "bureau")
    >>> sql = build_select(bureau, numeric_aggregations(bureau, ["AMT_ANNUITY"]), source="bureau")
    >>> 'WHERE "DAYS_CREDIT" < 0' in sql
    True
    """
    if source is None:
        source = _default_source(table)
    columns = ",\n       ".join(f'{spec.expression} AS "{spec.feature}"' for spec in specs)
    return (
        f'SELECT "{table.key_column}",\n       {columns}\n'
        f"FROM ({source}) AS source\n"
        f"WHERE {table.as_of_filter}\n"
        f'GROUP BY "{table.key_column}"'
    )


def _default_source(table: SatelliteTable) -> str:
    """Resolve a table's ``FROM`` expression, expanding a declared join.

    Parameters
    ----------
    table : SatelliteTable
        The table.

    Returns
    -------
    str
        A CSV reader, or the join that reaches ``SK_ID_CURR``.
    """
    if table.join_sql is None:
        return f"SELECT * FROM {csv_source(table.name)}"
    return table.join_sql.format(
        bureau_balance=csv_source("bureau_balance"), bureau=csv_source("bureau")
    )


def introspect(
    con: duckdb.DuckDBPyConnection, table: SatelliteTable, source: str | None = None
) -> tuple[list[str], dict[str, list[object]], list[str]]:
    """Read a table's column types and category levels.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection.
    table : SatelliteTable
        The table to inspect.
    source : str, optional
        ``FROM`` expression. Defaults as in :func:`build_select`.

    Returns
    -------
    numeric_columns : list of str
    levels : dict
        Categorical column to its observed levels, excluding columns with more
        than :data:`CATEGORY_LEVEL_CAP` of them.
    skipped : list of str
        Categorical columns excluded for cardinality, with their level counts.
    """
    if source is None:
        source = _default_source(table)
    described = con.execute(f"DESCRIBE SELECT * FROM ({source}) AS source LIMIT 0").df()
    numeric_columns = described.loc[
        described["column_type"].isin(["BIGINT", "INTEGER", "DOUBLE", "DECIMAL", "HUGEINT"]),
        "column_name",
    ].tolist()
    categorical_columns = described.loc[
        described["column_type"] == "VARCHAR", "column_name"
    ].tolist()

    levels: dict[str, list[object]] = {}
    skipped: list[str] = []
    for column in categorical_columns:
        if column in table.drop_columns:
            continue
        observed = (
            con.execute(
                f'SELECT DISTINCT "{column}" AS level FROM ({source}) AS source '
                f"WHERE {table.as_of_filter} ORDER BY 1"
            )
            .df()["level"]
            .tolist()
        )
        if len(observed) > CATEGORY_LEVEL_CAP:
            skipped.append(f"{column} ({len(observed)} levels)")
            continue
        levels[column] = observed
    return numeric_columns, levels, skipped


def manifest_frame(specs: list[FeatureSpec]) -> pd.DataFrame:
    """Turn feature specs into the auditable manifest.

    One row per feature, naming where it came from and under which restriction.
    A feature that cannot produce a manifest row cannot enter the frame — that
    invariant is what ``tests/test_features.py`` enforces.

    Parameters
    ----------
    specs : list of FeatureSpec
        Every feature in the frame.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``source_table``, ``source_column``,
        ``aggregation``, ``as_of_filter``.

    Examples
    --------
    >>> bureau = next(t for t in SATELLITES if t.name == "bureau")
    >>> manifest_frame(numeric_aggregations(bureau, ["AMT_ANNUITY"])).shape[1]
    5
    """
    return pd.DataFrame(
        [
            {
                "feature": spec.feature,
                "source_table": spec.source_table,
                "source_column": spec.source_column,
                "aggregation": spec.aggregation,
                "as_of_filter": spec.as_of_filter,
            }
            for spec in specs
        ]
    )


def filter_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Count what each as-of filter removes.

    Reported whether or not it removes anything. Home Credit's satellite tables
    are largely historic by construction, so several filters are expected to be
    no-ops — and a no-op that is measured is evidence, while a no-op that is
    assumed is a hole in the audit trail.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection.

    Returns
    -------
    pandas.DataFrame
        Columns: ``table``, ``as_of_filter``, ``counted_over``, ``rows_before``,
        ``rows_kept``, ``rows_removed``, ``share_removed``.

        ``counted_over`` matters for ``bureau_balance``, whose rows are counted
        after the join to an already-filtered ``bureau``. Its ``rows_before`` is
        therefore smaller than the raw file, and reading it as a raw-file count
        would overstate how much its own filter removed.
    """
    rows = []
    for table in SATELLITES:
        source = _default_source(table)
        before, kept = con.execute(
            f"SELECT COUNT(*), COUNT(*) FILTER (WHERE {table.as_of_filter}) "
            f"FROM ({source}) AS source"
        ).fetchone()
        rows.append(
            {
                "table": table.name,
                "as_of_filter": table.as_of_filter,
                "counted_over": (
                    "the raw table"
                    if table.join_sql is None
                    else "the table after the upstream join, itself already filtered"
                ),
                "rows_before": int(before),
                "rows_kept": int(kept),
                "rows_removed": int(before - kept),
                "share_removed": (before - kept) / before if before else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_frame(
    con: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Aggregate all six satellites and left-join them onto the applications.

    A left join, so an applicant with no bureau history keeps their row and
    acquires nulls. Dropping them would silently select for customers with a
    credit file, which is itself a decision with fairness consequences.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection.

    Returns
    -------
    frame : pandas.DataFrame
        One row per ``SK_ID_CURR``, application columns plus every aggregate, sorted
        by ``SK_ID_CURR``. The sort is not cosmetic: DuckDB does not guarantee a row
        order across parallel joins, and an unsorted frame produces a different
        stratified split on every build.
    manifest : pandas.DataFrame
        Provenance for every column, application columns included.
    skipped : list of str
        Categorical columns excluded for cardinality, tagged by table.
    """
    application = csv_source("application_train")
    replacements = ", ".join(
        f'NULLIF("{column}", {SENTINEL}) AS "{column}"' for column in APPLICATION_SENTINEL_COLUMNS
    )
    flags = ", ".join(
        f'CASE WHEN "{column}" = {SENTINEL} THEN 1 ELSE 0 END AS "{column}_IS_SENTINEL"'
        for column in APPLICATION_SENTINEL_COLUMNS
    )
    select = f"SELECT * REPLACE ({replacements}), {flags} FROM {application}"
    all_specs: list[FeatureSpec] = []
    skipped: list[str] = []

    for table in SATELLITES:
        numeric_columns, levels, table_skipped = introspect(con, table)
        specs = numeric_aggregations(table, numeric_columns) + categorical_aggregations(
            table, levels
        )
        all_specs.extend(specs)
        skipped.extend(f"{table.name}.{item}" for item in table_skipped)
        select = (
            f'SELECT base.*, agg.* EXCLUDE ("{table.key_column}")\n'
            f"FROM ({select}) AS base\n"
            f"LEFT JOIN ({build_select(table, specs)}) AS agg\n"
            f'ON base."SK_ID_CURR" = agg."{table.key_column}"'
        )

    # DuckDB parallelises these joins and does not promise a row order. Without an
    # explicit sort the frame comes back shuffled differently on every build, which
    # silently changes the stratified split's assignment and therefore every number
    # downstream. Ordering by the key makes the whole pipeline reproducible.
    frame = con.execute(f'SELECT * FROM ({select}) ORDER BY "SK_ID_CURR"').df()

    application_columns = con.execute(f"DESCRIBE SELECT * FROM {application} LIMIT 0").df()
    application_specs = [
        FeatureSpec(
            feature=column,
            source_table="application_train",
            source_column=column,
            aggregation=(
                f"none — application-level, NULLIF(column, {SENTINEL})"
                if column in APPLICATION_SENTINEL_COLUMNS
                else "none — application-level"
            ),
            expression=f'"{column}"',
            as_of_filter="none — the application is the as-of point",
        )
        for column in application_columns["column_name"]
    ]
    application_specs += [
        FeatureSpec(
            feature=f"{column}_IS_SENTINEL",
            source_table="application_train",
            source_column=column,
            aggregation=f"flag — column equals {SENTINEL}",
            expression=f'CASE WHEN "{column}" = {SENTINEL} THEN 1 ELSE 0 END',
            as_of_filter="none — the application is the as-of point",
        )
        for column in APPLICATION_SENTINEL_COLUMNS
    ]
    manifest = manifest_frame(application_specs + all_specs)
    return frame, manifest, skipped
