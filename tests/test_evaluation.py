"""Discrimination and calibration metrics, checked against arithmetic.

KS and Gini are the two ADIL computes itself; everything else comes from SPINE,
which verifies its own against scikit-learn in `reports/proof.json`. Checking
these against a four-row case that can be worked on paper keeps ADIL honest about
the two it owns.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from adil import evaluation


class TestKolmogorovSmirnov:
    def test_perfect_separation_is_one(self):
        assert evaluation.ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)

    def test_interleaved_case_matches_hand_computation(self):
        # Scores descending: 0.4(1), 0.3(0), 0.2(1), 0.1(0).
        # TPR-FPR at each cut: 0.5, 0.0, 0.5, 0.0  ->  KS = 0.5
        assert evaluation.ks_statistic([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.5)

    def test_a_useless_score_is_near_zero(self):
        assert evaluation.ks_statistic([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_is_bounded(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 500)
        p = rng.random(500)
        assert 0.0 <= evaluation.ks_statistic(y, p) <= 1.0


class TestGini:
    def test_is_twice_auc_minus_one(self):
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.3, 500)
        p = rng.random(500)
        assert evaluation.gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)

    def test_perfect_ranking_is_one(self):
        assert evaluation.gini([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)

    def test_reversed_ranking_is_minus_one(self):
        assert evaluation.gini([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(-1.0)


class TestMetricSet:
    @pytest.fixture
    def scored(self):
        rng = np.random.default_rng(2)
        y = rng.binomial(1, 0.08, 4000)
        p = np.clip(0.08 + 0.25 * (y - 0.08) + rng.normal(0, 0.05, 4000), 1e-6, 1 - 1e-6)
        return y, p

    def test_reports_every_metric_the_headline_table_needs(self, scored):
        y, p = scored
        result = evaluation.metric_set(y, p, split="test")
        for key in ("split", "n", "event_rate", "pr_auc", "auc", "ks", "gini", "brier", "ece"):
            assert key in result

    def test_carries_the_split_it_was_computed_on(self, scored):
        # Every metric reported alongside its split is a project convention, so a
        # metric set that cannot name its split should not be constructible.
        y, p = scored
        assert evaluation.metric_set(y, p, split="calibration")["split"] == "calibration"

    def test_values_are_plain_floats_not_numpy_scalars(self, scored):
        # These land in JSON. numpy scalars do not serialise.
        y, p = scored
        result = evaluation.metric_set(y, p, split="test")
        assert isinstance(result["pr_auc"], float)
        assert isinstance(result["n"], int)


class TestBootstrapInterval:
    @pytest.fixture
    def scored(self):
        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.1, 3000)
        p = np.clip(0.1 + 0.3 * (y - 0.1) + rng.normal(0, 0.08, 3000), 1e-6, 1 - 1e-6)
        return y, p

    def test_interval_brackets_the_point_estimate(self, scored):
        y, p = scored
        low, high = evaluation.bootstrap_interval(y, p, roc_auc_score, draws=200)
        assert low < roc_auc_score(y, p) < high

    def test_is_reproducible_from_the_seed(self, scored):
        y, p = scored
        first = evaluation.bootstrap_interval(y, p, roc_auc_score, draws=100, seed=11)
        second = evaluation.bootstrap_interval(y, p, roc_auc_score, draws=100, seed=11)
        assert first == second

    def test_more_draws_do_not_change_the_answer_much(self, scored):
        y, p = scored
        few = evaluation.bootstrap_interval(y, p, roc_auc_score, draws=100, seed=5)
        many = evaluation.bootstrap_interval(y, p, roc_auc_score, draws=600, seed=5)
        assert abs(few[0] - many[0]) < 0.02
        assert abs(few[1] - many[1]) < 0.02
