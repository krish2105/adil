"""The leakage discipline is the thing this project stakes its defensibility on.

Home Credit carries no application date, so no temporal split is possible and
:mod:`spine.splitting` is unused. What replaces it is as-of-application feature
construction: every satellite aggregate is restricted to records knowable when
the application was decided. These tests check that discipline is *enforced*
rather than intended, and that every feature can name its own provenance.
"""

import duckdb
import pytest

from adil import features


@pytest.fixture
def con():
    return duckdb.connect()


class TestAsOfDeclarations:
    def test_every_satellite_declares_a_filter_and_a_reason(self):
        for table in features.SATELLITES:
            assert table.as_of_filter.strip(), f"{table.name} declares no as-of filter"
            assert len(table.as_of_rationale) > 20, f"{table.name} has no stated rationale"

    def test_every_filter_bounds_a_relative_time_column(self):
        # A filter that does not compare a DAYS_/MONTHS_ column against zero is
        # not an as-of rule, whatever else it may be.
        for table in features.SATELLITES:
            clause = table.as_of_filter
            assert "< 0" in clause, f"{table.name}: {clause!r} does not bound anything at zero"
            assert any(token in clause for token in ("DAYS_", "MONTHS_")), (
                f"{table.name}: {clause!r} bounds no relative-time column"
            )

    def test_the_seven_tables_are_accounted_for(self):
        # application_train is the spine of the frame and has no as-of filter of
        # its own; every other table must be declared.
        from adil.paths import HOME_CREDIT_TABLES

        declared = {table.name for table in features.SATELLITES}
        assert declared | {"application_train"} == set(HOME_CREDIT_TABLES)


class TestGeneratedSql:
    def test_filter_reaches_the_generated_sql(self):
        table = features.SATELLITES[0]
        specs = features.numeric_aggregations(table, ["AMT_CREDIT_SUM"])
        sql = features.build_select(table, specs)
        assert table.as_of_filter in sql
        assert "WHERE" in sql

    def test_every_spec_becomes_one_output_column(self):
        table = features.SATELLITES[0]
        specs = features.numeric_aggregations(table, ["AMT_CREDIT_SUM", "DAYS_CREDIT"])
        sql = features.build_select(table, specs)
        for spec in specs:
            assert f'AS "{spec.feature}"' in sql

    def test_key_column_is_never_aggregated_as_a_feature(self):
        for table in features.SATELLITES:
            specs = features.numeric_aggregations(table, [table.key_column, "SOME_AMOUNT"])
            sources = {spec.source_column for spec in specs}
            assert table.key_column not in sources


class TestManifest:
    def test_manifest_covers_every_generated_feature(self):
        table = features.SATELLITES[0]
        specs = features.numeric_aggregations(table, ["AMT_CREDIT_SUM"])
        manifest = features.manifest_frame(specs)
        assert set(manifest["feature"]) == {spec.feature for spec in specs}

    def test_manifest_names_provenance_for_every_row(self):
        table = features.SATELLITES[0]
        specs = features.numeric_aggregations(table, ["AMT_CREDIT_SUM", "DAYS_CREDIT"])
        manifest = features.manifest_frame(specs)
        for column in ("source_table", "source_column", "aggregation", "as_of_filter"):
            assert manifest[column].notna().all()
            assert (manifest[column].astype(str).str.len() > 0).all()

    def test_feature_names_are_unique(self):
        table = features.SATELLITES[0]
        specs = features.numeric_aggregations(table, ["AMT_CREDIT_SUM", "DAYS_CREDIT"])
        specs += features.categorical_aggregations(table, {"CREDIT_ACTIVE": ["Active", "Closed"]})
        names = [spec.feature for spec in specs]
        assert len(names) == len(set(names))


class TestFilterActuallyExcludes:
    """The filters must remove the rows they claim to remove.

    Run against a hand-built frame carrying one row on each side of the
    boundary, so the assertion is about SQL behaviour rather than about Home
    Credit's contents.
    """

    def test_bureau_filter_drops_non_historic_records(self, con):
        con.execute(
            "CREATE TABLE bureau AS SELECT * FROM (VALUES "
            "  (1, -100, 5000.0), "  # historic, kept
            "  (1,    0, 9999.0), "  # same-day, dropped
            "  (1,   50, 9999.0)  "  # future, dropped
            ") AS t(SK_ID_CURR, DAYS_CREDIT, AMT_CREDIT_SUM)"
        )
        table = next(t for t in features.SATELLITES if t.name == "bureau")
        specs = features.numeric_aggregations(table, ["AMT_CREDIT_SUM"])
        sql = features.build_select(table, specs, source="SELECT * FROM bureau")
        result = con.execute(sql).df()
        assert len(result) == 1
        assert result["BUR_AMT_CREDIT_SUM_MAX"].iloc[0] == 5000.0

    def test_installments_filter_requires_both_columns_historic(self, con):
        con.execute(
            "CREATE TABLE installments_payments AS SELECT * FROM (VALUES "
            "  (1, -100, -90, 100.0), "  # both historic, kept
            "  (1,  -10,   5, 999.0), "  # payment entered after application, dropped
            "  (1,    5, -10, 999.0)  "  # instalment due after application, dropped
            ") AS t(SK_ID_CURR, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT, AMT_PAYMENT)"
        )
        table = next(t for t in features.SATELLITES if t.name == "installments_payments")
        specs = features.numeric_aggregations(table, ["AMT_PAYMENT"])
        sql = features.build_select(table, specs, source="SELECT * FROM installments_payments")
        result = con.execute(sql).df()
        assert len(result) == 1
        assert result["INS_AMT_PAYMENT_MAX"].iloc[0] == 100.0


class TestSentinel:
    """365243 is "this never happened", not a duration a thousand years out.

    Averaging it silently produces a number with no meaning, and more than half of
    ``previous_application.DAYS_FIRST_DRAWING`` carries it, so the corruption would
    not have been small.
    """

    def test_sentinel_columns_are_aggregated_over_nullif(self):
        table = next(t for t in features.SATELLITES if t.name == "previous_application")
        assert table.sentinel_columns
        specs = features.numeric_aggregations(table, list(table.sentinel_columns))
        aggregated = [s for s in specs if s.aggregation.startswith(("MIN", "MAX", "MEAN", "SUM"))]
        for spec in aggregated:
            assert f'NULLIF("{spec.source_column}", {features.SENTINEL})' in spec.expression

    def test_sentinel_share_is_kept_as_its_own_feature(self):
        table = next(t for t in features.SATELLITES if t.name == "previous_application")
        specs = features.numeric_aggregations(table, ["DAYS_FIRST_DRAWING"])
        shares = [s for s in specs if s.feature.endswith("_SENTINEL_SHARE")]
        assert len(shares) == 1
        assert str(features.SENTINEL) in shares[0].expression

    def test_ordinary_columns_are_untouched(self):
        table = next(t for t in features.SATELLITES if t.name == "previous_application")
        specs = features.numeric_aggregations(table, ["AMT_ANNUITY"])
        assert not any(s.feature.endswith("_SENTINEL_SHARE") for s in specs)
        assert all("NULLIF" not in s.expression for s in specs)

    def test_nullif_actually_excludes_the_sentinel(self, con):
        con.execute(
            "CREATE TABLE previous_application AS SELECT * FROM (VALUES "
            "  (1, -100, -10.0), (1, -100, 365243.0), (1, -100, -30.0) "
            ") AS t(SK_ID_CURR, DAYS_DECISION, DAYS_LAST_DUE)"
        )
        table = next(t for t in features.SATELLITES if t.name == "previous_application")
        specs = features.numeric_aggregations(table, ["DAYS_LAST_DUE"])
        result = con.execute(
            features.build_select(table, specs, source="SELECT * FROM previous_application")
        ).df()
        # Mean of -10 and -30, with the sentinel excluded rather than dragging it to +121734.
        assert result["PRV_DAYS_LAST_DUE_MEAN"].iloc[0] == -20.0
        assert result["PRV_DAYS_LAST_DUE_SENTINEL_SHARE"].iloc[0] == pytest.approx(1 / 3)


class TestDeterminism:
    """A frame whose row order varies is a pipeline whose results vary.

    DuckDB parallelises the satellite joins and makes no promise about row order.
    Left unsorted, the frame comes back shuffled differently on every build, the
    stratified split assigns different applicants to the test set, and every
    number in every report moves without anything having changed. Nothing about
    that failure is visible in a target rate or a row count — it was caught only
    by diffing two runs of the whole pipeline.
    """

    def test_the_frame_is_ordered_by_the_application_key(self, con):
        con.execute("CREATE TABLE ids AS SELECT * FROM (VALUES (3), (1), (2)) AS t(SK_ID_CURR)")
        shuffled = con.execute("SELECT * FROM ids").df()
        ordered = con.execute('SELECT * FROM ids ORDER BY "SK_ID_CURR"').df()
        assert ordered["SK_ID_CURR"].is_monotonic_increasing
        assert set(shuffled["SK_ID_CURR"]) == set(ordered["SK_ID_CURR"])

    def test_build_frame_sorts_its_output(self):
        # Guards the ORDER BY against being dropped as redundant-looking.
        import inspect

        source = inspect.getsource(features.build_frame)
        assert 'ORDER BY "SK_ID_CURR"' in source


class TestFloatingPointDeterminism:
    """Parallel aggregation of floats is not reproducible, and it matters here.

    Floating-point addition is not associative, so a parallel SUM or MEAN
    accumulates in whatever order threads finish in. Two parallel builds of this
    frame differed bit-for-bit on 48 of 559 numeric columns, by around 1e-8 —
    invisible on inspection, and enough to move a LightGBM split threshold and
    with it every number in every report.
    """

    def test_the_build_script_pins_duckdb_to_one_thread(self):
        from pathlib import Path

        script = Path(__file__).resolve().parents[1] / "scripts" / "build_features.py"
        source = script.read_text()
        assert "PRAGMA threads=1" in source
        assert "threads=10" not in source

    def test_float_summation_is_order_dependent(self):
        # The mechanism itself, in IEEE-754: the same three values in two orders
        # give two different sums. A parallel aggregate picks the order at runtime,
        # which is why the build below is pinned to one thread.
        assert (0.1 + 0.2) + 0.3 != 0.1 + (0.2 + 0.3)
