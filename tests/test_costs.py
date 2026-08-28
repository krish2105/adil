"""The cost model behind the approval cutoff.

Every number in notebook 06 depends on a cost matrix, so the arithmetic is pinned
here and the one published ratio available to this project is kept as a constant
rather than a number someone typed into a notebook.
"""

import numpy as np
import pytest

from adil import costs


class TestCostMatrix:
    def test_layout_matches_what_spine_expects(self):
        # spine.decisions indexes [true label][predicted label] and predicts the
        # event, which here is default. Predicting the event means declining.
        matrix = costs.cost_matrix(5.0)
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 0.0, "approving an applicant who repays is the baseline"
        assert matrix[0, 1] == 1.0, "declining someone who would have repaid costs the margin"
        assert matrix[1, 0] == 5.0, "approving someone who defaults costs the loss"
        assert matrix[1, 1] == 0.0, "declining someone who defaults costs nothing"

    def test_ratio_scales_only_the_loss(self):
        assert costs.cost_matrix(1.0)[1, 0] == 1.0
        assert costs.cost_matrix(20.0)[1, 0] == 20.0

    def test_rejects_a_non_positive_ratio(self):
        with pytest.raises(ValueError, match="positive"):
            costs.cost_matrix(0.0)

    def test_from_amounts_reduces_to_the_ratio(self):
        # A loss of 4,500 against a forgone margin of 900 is a 5:1 problem, and only
        # the ratio can move a threshold — the scale moves the total, not the cutoff.
        from_amounts = costs.cost_matrix_from_amounts(loss=4500.0, margin=900.0)
        assert from_amounts[1, 0] / from_amounts[0, 1] == pytest.approx(5.0)


class TestPublishedRatio:
    def test_the_german_credit_ratio_is_recorded_with_its_source(self):
        # The only cost ratio in this project that was published rather than assumed.
        assert costs.GERMAN_CREDIT_COST_RATIO == 5.0
        assert "german" in costs.GERMAN_CREDIT_SOURCE.lower()

    def test_the_scenario_caveat_says_it_is_not_a_finding(self):
        assert "SCENARIO" in costs.SCENARIO_CAVEAT
        assert "not a finding" in costs.SCENARIO_CAVEAT


class TestDecisionCosts:
    @pytest.fixture
    def scored(self):
        # Four applicants: two safe and repaying, one risky and defaulting, one
        # risky but repaying.
        y = np.array([0, 0, 1, 0])
        p = np.array([0.05, 0.10, 0.80, 0.70])
        return y, p

    def test_approves_below_the_threshold(self, scored):
        y, p = scored
        result = costs.decision_costs(y, p, threshold=0.5, cost_matrix=costs.cost_matrix(5.0))
        assert result["n_approved"] == 2
        assert result["approval_rate"] == pytest.approx(0.5)

    def test_totals_the_two_error_types(self, scored):
        y, p = scored
        result = costs.decision_costs(y, p, threshold=0.5, cost_matrix=costs.cost_matrix(5.0))
        # Declined: the defaulter (correct, no cost) and one good applicant (costs 1).
        assert result["n_declined_good"] == 1
        assert result["n_approved_bad"] == 0
        assert result["total_cost"] == pytest.approx(1.0)

    def test_cost_per_approved_divides_by_approvals_not_applications(self, scored):
        y, p = scored
        result = costs.decision_costs(y, p, threshold=0.5, cost_matrix=costs.cost_matrix(5.0))
        assert result["cost_per_approved"] == pytest.approx(1.0 / 2)
        assert result["cost_per_application"] == pytest.approx(1.0 / 4)

    def test_approving_a_defaulter_costs_the_loss(self, scored):
        y, p = scored
        result = costs.decision_costs(y, p, threshold=0.95, cost_matrix=costs.cost_matrix(5.0))
        assert result["n_approved"] == 4
        assert result["n_approved_bad"] == 1
        assert result["total_cost"] == pytest.approx(5.0)

    def test_bad_rate_among_approved_is_reported(self, scored):
        y, p = scored
        result = costs.decision_costs(y, p, threshold=0.95, cost_matrix=costs.cost_matrix(5.0))
        assert result["bad_rate_among_approved"] == pytest.approx(0.25)

    def test_approving_nobody_gives_nan_rather_than_a_division_error(self, scored):
        y, p = scored
        result = costs.decision_costs(y, p, threshold=0.0, cost_matrix=costs.cost_matrix(5.0))
        assert result["n_approved"] == 0
        assert np.isnan(result["cost_per_approved"])
        assert np.isnan(result["bad_rate_among_approved"])


class TestAnalyticThreshold:
    def test_matches_the_closed_form(self):
        # For a calibrated model the cost-minimising cutoff is co / (co + cu).
        assert costs.analytic_threshold(costs.cost_matrix(5.0)) == pytest.approx(1 / 6)
        assert costs.analytic_threshold(costs.cost_matrix(1.0)) == pytest.approx(0.5)

    def test_a_costlier_default_lowers_the_cutoff(self):
        assert costs.analytic_threshold(costs.cost_matrix(20.0)) < costs.analytic_threshold(
            costs.cost_matrix(2.0)
        )
