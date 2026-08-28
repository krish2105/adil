"""The challenger's configuration, and the property that makes rung R3 mean anything.

The interesting test here is the monotonicity one. A declared constraint that
LightGBM quietly ignored would make R3 a rung that costs nothing because it does
nothing, and the reported cost of monotonicity would be a fiction. So the test
builds data whose true relationship runs the wrong way, constrains against it, and
checks the fitted model actually obeys.
"""

import numpy as np
import pandas as pd
import pytest

from adil import challenger


class TestRungs:
    def test_names_are_unique_and_ordered(self):
        names = [rung.name for rung in challenger.RUNGS]
        assert names == sorted(names)
        assert len(names) == len(set(names))

    def test_every_rung_describes_what_it_adds(self):
        for rung in challenger.RUNGS:
            assert len(rung.description) > 20

    def test_the_ladder_adds_one_constraint_at_a_time(self):
        by_name = {rung.name: rung for rung in challenger.RUNGS}
        assert not by_name["R1"].monotone and by_name["R1"].feature_cap is None
        assert by_name["R3"].monotone and by_name["R3"].feature_cap is None
        assert by_name["R4"].monotone and by_name["R4"].feature_cap is not None

    def test_the_seed_is_recorded_in_the_parameters(self):
        assert isinstance(challenger.BASE_PARAMS["seed"], int)


class TestDesignMatrix:
    def test_object_columns_become_categorical(self):
        frame = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})
        design = challenger.design_matrix(frame, ["a", "b"])
        assert isinstance(design["a"].dtype, pd.CategoricalDtype)

    def test_nullable_extension_dtypes_become_float(self):
        # LightGBM cannot consume pandas' nullable Int64 or boolean dtypes, which
        # arrive from parquet. Left alone they raise at fit time.
        frame = pd.DataFrame(
            {"i": pd.array([1, None], dtype="Int64"), "b": pd.array([True, None], dtype="boolean")}
        )
        design = challenger.design_matrix(frame, ["i", "b"])
        assert design["i"].dtype == np.float64
        assert design["b"].dtype == np.float64
        assert np.isnan(design["i"].iloc[1])

    def test_column_order_follows_the_feature_list(self):
        frame = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        assert challenger.design_matrix(frame, ["c", "a"]).columns.tolist() == ["c", "a"]

    def test_missing_feature_is_an_error_not_a_silent_drop(self):
        frame = pd.DataFrame({"a": [1.0]})
        with pytest.raises(KeyError, match="absent"):
            challenger.design_matrix(frame, ["a", "nope"])


class TestMonotonicity:
    """A declared constraint has to bind, or R3 measures nothing."""

    @pytest.fixture
    def contrary(self):
        # y falls as x rises. A +1 constraint forbids the model from following that,
        # so a constrained fit must come out flat or rising, never falling.
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 4000)
        y = rng.binomial(1, np.clip(0.9 - 0.8 * x, 0.01, 0.99))
        return pd.DataFrame({"x": x, "noise": rng.normal(size=4000)}), y

    def test_unconstrained_fit_follows_the_data_downwards(self, contrary):
        frame, y = contrary
        model = challenger.fit(frame, y, ["x", "noise"], monotone=None, num_boost_round=60)
        grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 25), "noise": 0.0})
        predicted = model.predict(grid)
        assert predicted[0] > predicted[-1], "the unconstrained model should track the data"

    def test_constrained_fit_never_decreases(self, contrary):
        frame, y = contrary
        model = challenger.fit(frame, y, ["x", "noise"], monotone=[1, 0], num_boost_round=60)
        grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 25), "noise": 0.0})
        predicted = model.predict(grid)
        assert np.all(np.diff(predicted) >= -1e-12), "the +1 constraint did not bind"

    def test_a_minus_one_constraint_never_increases(self, contrary):
        frame, y = contrary
        model = challenger.fit(frame, y, ["x", "noise"], monotone=[-1, 0], num_boost_round=60)
        grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 25), "noise": 0.0})
        predicted = model.predict(grid)
        assert np.all(np.diff(predicted) <= 1e-12)


class TestGainImportance:
    def test_covers_every_feature_and_sorts_descending(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 2000)
        frame = pd.DataFrame({"signal": x, "noise": rng.normal(size=2000)})
        y = rng.binomial(1, np.clip(0.1 + 0.8 * x, 0.01, 0.99))
        model = challenger.fit(frame, y, ["signal", "noise"], monotone=None, num_boost_round=40)
        importance = challenger.gain_importance(model, ["signal", "noise"])
        assert set(importance.index) == {"signal", "noise"}
        assert importance.is_monotonic_decreasing
        assert importance.idxmax() == "signal"


class TestDeterminism:
    """A fixed seed is not by itself enough to make LightGBM reproducible.

    LightGBM builds histograms across threads and chooses its row-wise or
    col-wise strategy from a runtime heuristic, so two fits on identical data can
    differ in the sixth decimal place. That is small enough to look like nothing
    and large enough to move every number in every report, which is exactly the
    kind of drift that is discovered late and explained badly.
    """

    def test_the_determinism_flags_are_set(self):
        assert challenger.BASE_PARAMS["deterministic"] is True
        assert challenger.BASE_PARAMS["force_row_wise"] is True

    def test_two_fits_on_the_same_data_agree_exactly(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(
            {"a": rng.normal(size=3000), "b": rng.normal(size=3000), "c": rng.normal(size=3000)}
        )
        y = rng.binomial(1, 1 / (1 + np.exp(-frame["a"])))
        first = challenger.fit(frame, y, ["a", "b", "c"], num_boost_round=50)
        second = challenger.fit(frame, y, ["a", "b", "c"], num_boost_round=50)
        assert np.array_equal(first.predict(frame), second.predict(frame))
