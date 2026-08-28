"""Home Credit has no time axis, so the split is stratified rather than temporal.

That is a documented exception, not a convenience, and it carries obligations:
the splits must be disjoint, exhaustive, reproducible from a recorded seed, and
close enough in target rate that a metric difference between them is a model
result rather than a sampling artifact.
"""

import numpy as np
import pandas as pd
import pytest

from adil import split


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 10_000
    return pd.DataFrame(
        {"SK_ID_CURR": np.arange(100_000, 100_000 + n), "TARGET": rng.binomial(1, 0.08, n)}
    )


class TestAssignment:
    def test_every_row_lands_in_exactly_one_split(self, frame):
        assigned = split.stratified_split(frame)
        assert len(assigned) == len(frame)
        assert set(assigned["split"]) == {"train", "calibration", "test"}

    def test_splits_share_no_identifiers(self, frame):
        assigned = split.stratified_split(frame)
        groups = {name: set(part["SK_ID_CURR"]) for name, part in assigned.groupby("split")}
        assert not groups["train"] & groups["calibration"]
        assert not groups["train"] & groups["test"]
        assert not groups["calibration"] & groups["test"]

    def test_proportions_are_honoured(self, frame):
        assigned = split.stratified_split(frame)
        shares = assigned["split"].value_counts(normalize=True)
        assert shares["train"] == pytest.approx(0.60, abs=0.01)
        assert shares["calibration"] == pytest.approx(0.20, abs=0.01)
        assert shares["test"] == pytest.approx(0.20, abs=0.01)


class TestStratification:
    def test_target_rates_agree_within_a_tenth_of_a_point(self, frame):
        assigned = split.stratified_split(frame)
        rates = assigned.groupby("split")["TARGET"].mean()
        assert rates.max() - rates.min() < 0.001


class TestReproducibility:
    def test_same_seed_gives_the_same_assignment(self, frame):
        first = split.stratified_split(frame, seed=7)
        second = split.stratified_split(frame, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_a_different_seed_gives_a_different_assignment(self, frame):
        first = split.stratified_split(frame, seed=7)
        second = split.stratified_split(frame, seed=8)
        assert not first["split"].equals(second["split"])

    def test_the_default_seed_is_recorded_not_incidental(self):
        assert isinstance(split.SEED, int)
