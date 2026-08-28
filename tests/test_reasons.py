"""Adverse-action reason codes, and the gate that decides whether they are usable.

Rung R5 is a pass/fail gate rather than a model change: it asks whether the top-3
reasons given to a declined applicant survive a small perturbation of their file.
The threshold is registered in `adil.reasons` before any flip rate is measured,
and these tests pin the mechanics so the measurement itself cannot drift.
"""

import numpy as np
import pytest

from adil import reasons


class TestTopReasons:
    def test_picks_the_largest_positive_contributions(self):
        # Row 0: feature b then a push risk up; c pushes it down and is not a reason.
        matrix = np.array([[0.2, 0.5, -0.9], [0.7, 0.1, 0.3]])
        chosen = reasons.top_reasons(matrix, ["a", "b", "c"], k=2)
        assert chosen[0] == ("b", "a")
        assert chosen[1] == ("a", "c")

    def test_a_reason_must_push_risk_upwards(self):
        # Nothing raised this applicant's risk, so there is no adverse reason to give.
        matrix = np.array([[-0.1, -0.4, -0.2]])
        assert reasons.top_reasons(matrix, ["a", "b", "c"], k=3) == [()]

    def test_returns_fewer_than_k_when_fewer_contributions_are_positive(self):
        matrix = np.array([[0.3, -0.4, -0.2]])
        assert reasons.top_reasons(matrix, ["a", "b", "c"], k=3) == [("a",)]

    def test_ties_break_deterministically(self):
        matrix = np.array([[0.5, 0.5, 0.1]])
        first = reasons.top_reasons(matrix, ["a", "b", "c"], k=2)
        second = reasons.top_reasons(matrix, ["a", "b", "c"], k=2)
        assert first == second

    def test_rejects_a_name_list_that_does_not_match_the_matrix(self):
        with pytest.raises(ValueError, match="columns"):
            reasons.top_reasons(np.zeros((2, 3)), ["a", "b"], k=2)


class TestFlipRate:
    def test_identical_reasons_do_not_flip(self):
        base = [("a", "b", "c"), ("d", "e", "f")]
        assert reasons.flip_rate(base, base) == 0.0

    def test_completely_different_reasons_all_flip(self):
        base = [("a", "b", "c")]
        moved = [("x", "y", "z")]
        assert reasons.flip_rate(base, moved) == 1.0

    def test_reordering_is_not_a_flip(self):
        # The applicant is told the same three things. Which is printed first is a
        # presentation choice, not a different explanation.
        base = [("a", "b", "c")]
        reordered = [("c", "a", "b")]
        assert reasons.flip_rate(base, reordered) == 0.0

    def test_one_substitution_counts_as_a_flip(self):
        base = [("a", "b", "c"), ("a", "b", "c")]
        moved = [("a", "b", "c"), ("a", "b", "z")]
        assert reasons.flip_rate(base, moved) == 0.5

    def test_mismatched_lengths_are_an_error(self):
        with pytest.raises(ValueError, match="same length"):
            reasons.flip_rate([("a",)], [("a",), ("b",)])


class TestRegisteredGate:
    """The gate's value depends entirely on it having been fixed in advance."""

    def test_the_threshold_and_perturbation_are_declared_constants(self):
        assert 0 < reasons.FLIP_RATE_GATE < 1
        assert reasons.PERTURBATION_SIGMA > 0
        assert reasons.TOP_K >= 1

    def test_the_registration_states_the_threshold_it_registers(self):
        # The prose must carry the actual number, so the report cannot quote a
        # registration that says something different from what the code enforces.
        assert len(reasons.GATE_REGISTRATION) > 100
        assert f"{reasons.FLIP_RATE_GATE:.0%}" in reasons.GATE_REGISTRATION
        assert str(reasons.PERTURBATION_SIGMA) in reasons.GATE_REGISTRATION
        assert str(reasons.TOP_K) in reasons.GATE_REGISTRATION

    def test_verdict_is_a_plain_comparison(self):
        assert reasons.gate_verdict(reasons.FLIP_RATE_GATE - 0.01) == "pass"
        assert reasons.gate_verdict(reasons.FLIP_RATE_GATE + 0.01) == "fail"

    def test_exactly_at_the_threshold_passes(self):
        # "at most" is registered, so the boundary is inclusive; deciding this after
        # seeing the number would be the whole problem the gate exists to prevent.
        assert reasons.gate_verdict(reasons.FLIP_RATE_GATE) == "pass"


class TestPerturbation:
    def test_perturbs_numeric_columns_by_the_requested_scale(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"x": rng.normal(100, 10, 5000), "label": ["a"] * 5000})
        moved = reasons.perturb(frame, ["x", "label"], sigma=0.5, seed=1)
        shift = (moved["x"] - frame["x"]).std()
        assert shift == pytest.approx(0.5 * frame["x"].std(), rel=0.1)

    def test_leaves_categorical_columns_alone(self):
        import pandas as pd

        frame = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
        moved = reasons.perturb(frame, ["x", "label"], sigma=0.5, seed=1)
        assert moved["label"].tolist() == ["a", "b"]

    def test_is_reproducible_from_the_seed(self):
        import pandas as pd

        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        first = reasons.perturb(frame, ["x"], sigma=0.3, seed=7)
        second = reasons.perturb(frame, ["x"], sigma=0.3, seed=7)
        assert first["x"].tolist() == second["x"].tolist()

    def test_preserves_missingness(self):
        import pandas as pd

        frame = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        moved = reasons.perturb(frame, ["x"], sigma=0.5, seed=1)
        assert bool(np.isnan(moved["x"].iloc[1]))
