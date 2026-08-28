"""Weight of evidence and information value, checked against arithmetic.

`adil.scorecard` carries its own WoE and IV implementation, which exists so that
optbinning can be verified against the formula rather than against itself. These
tests pin that implementation to hand-computed values; notebook 02 then runs the
same comparison against optbinning's own binning table on a real feature.

The worked case, small enough to check on paper:

======  ======  =========  ==========  =============
Bin     Events  Nonevents  % of event  % of nonevent
======  ======  =========  ==========  =============
A           20         80    20/30      80/270
B           10        190    10/30     190/270
======  ======  =========  ==========  =============
"""

import numpy as np
import pandas as pd
import pytest

from adil import scorecard


@pytest.fixture
def worked_case():
    return pd.DataFrame({"bin": ["A", "B"], "n_event": [20, 10], "n_nonevent": [80, 190]})


class TestWeightOfEvidence:
    def test_matches_hand_computation(self):
        assert scorecard.weight_of_evidence(20, 80, 30, 270) == pytest.approx(
            -0.8109302162, abs=1e-10
        )
        assert scorecard.weight_of_evidence(10, 190, 30, 270) == pytest.approx(
            0.7472144018, abs=1e-10
        )

    def test_is_zero_when_a_bin_matches_the_population(self):
        # A bin holding exactly the population's odds carries no evidence.
        assert scorecard.weight_of_evidence(15, 135, 30, 270) == pytest.approx(0.0, abs=1e-12)

    def test_sign_convention_is_the_credit_one(self):
        # WoE = ln(%nonevent / %event): a bin richer in good accounts scores positive.
        risky = scorecard.weight_of_evidence(25, 50, 30, 270)
        safe = scorecard.weight_of_evidence(5, 220, 30, 270)
        assert risky < 0 < safe

    def test_an_empty_cell_gives_nan_not_a_number(self):
        # Zero events makes the log undefined. NaN says "this sample cannot answer
        # the question"; a smoothed value would say "this bin is very safe", and the
        # two call for different responses.
        assert np.isnan(scorecard.weight_of_evidence(0, 80, 30, 270))
        assert np.isnan(scorecard.weight_of_evidence(20, 0, 30, 270))


class TestInformationValue:
    def test_matches_hand_computation(self, worked_case):
        table = scorecard.bin_statistics(worked_case)
        assert table["iv_contribution"].iloc[0] == pytest.approx(0.3003445245, abs=1e-9)
        assert table["iv_contribution"].iloc[1] == pytest.approx(0.2767460748, abs=1e-9)
        assert scorecard.information_value(worked_case) == pytest.approx(0.5770905993, abs=1e-9)

    def test_is_zero_for_a_feature_that_separates_nothing(self):
        flat = pd.DataFrame({"bin": ["A", "B"], "n_event": [15, 15], "n_nonevent": [135, 135]})
        assert scorecard.information_value(flat) == pytest.approx(0.0, abs=1e-12)

    def test_is_never_negative(self, worked_case):
        assert scorecard.information_value(worked_case) > 0

    def test_bin_statistics_reports_the_observed_rate(self, worked_case):
        table = scorecard.bin_statistics(worked_case)
        assert table["event_rate"].tolist() == pytest.approx([0.2, 0.05])
        assert table["n"].tolist() == [100, 200]


class TestCandidateFeatures:
    def test_protected_attributes_are_excluded(self):
        frame = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3, 4],
                "TARGET": [0, 1, 0, 1],
                "CODE_GENDER": ["M", "F", "M", "F"],
                "DAYS_BIRTH": [-1, -2, -3, -4],
                "AMT_CREDIT": [10, 20, 30, 40],
            }
        )
        candidates = scorecard.candidate_features(frame)
        assert "AMT_CREDIT" in candidates
        for excluded in ("CODE_GENDER", "DAYS_BIRTH", "SK_ID_CURR", "TARGET"):
            assert excluded not in candidates

    def test_constant_columns_are_excluded(self):
        frame = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3, 4],
                "TARGET": [0, 1, 0, 1],
                "AMT_CREDIT": [10, 20, 30, 40],
                "ALWAYS_SEVEN": [7, 7, 7, 7],
                "ALL_NULL": [None, None, None, None],
            }
        )
        candidates = scorecard.candidate_features(frame)
        assert candidates == ["AMT_CREDIT"]

    def test_the_protected_list_is_declared_not_inferred(self):
        assert set(scorecard.PROTECTED_ATTRIBUTES) == {"CODE_GENDER", "DAYS_BIRTH"}


class TestPointsScaling:
    """PDO scaling is arithmetic and is checked as arithmetic.

    With 20 points to double the odds and 600 points set at 50:1, doubling the
    odds to 100:1 must move the score by exactly 20 points.
    """

    def test_factor_and_offset_match_the_formula(self):
        factor, offset = scorecard.pdo_scaling(pdo=20, base_odds=50, base_points=600)
        assert factor == pytest.approx(20 / np.log(2), abs=1e-12)
        assert offset == pytest.approx(600 - factor * np.log(50), abs=1e-12)

    def test_doubling_the_odds_moves_the_score_by_the_pdo(self):
        factor, offset = scorecard.pdo_scaling(pdo=20, base_odds=50, base_points=600)
        at_base = offset + factor * np.log(50)
        at_double = offset + factor * np.log(100)
        assert at_base == pytest.approx(600, abs=1e-9)
        assert at_double - at_base == pytest.approx(20, abs=1e-9)


class TestSelectCharacteristics:
    """Selection is greedy by IV with correlation pruning, and it logs its reasons.

    The log matters as much as the list: "why is this characteristic not in the
    scorecard?" is a question a model validator asks, and it should have an
    answer per row rather than an answer per model.
    """

    @pytest.fixture
    def ivs(self):
        return pd.Series({"strong": 0.30, "twin": 0.25, "independent": 0.10, "weak": 0.005})

    @pytest.fixture
    def corr(self):
        names = ["strong", "twin", "independent", "weak"]
        matrix = np.eye(4)
        matrix[0, 1] = matrix[1, 0] = 0.92
        return pd.DataFrame(matrix, index=names, columns=names)

    def test_strongest_is_kept_and_its_twin_is_dropped(self, ivs, corr):
        log = scorecard.select_characteristics(ivs, corr)
        chosen = log.loc[log["selected"], "feature"].tolist()
        assert chosen == ["strong", "independent"]

    def test_every_rejection_states_a_reason(self, ivs, corr):
        log = scorecard.select_characteristics(ivs, corr)
        rejected = log.loc[~log["selected"]]
        assert len(rejected) == 2
        assert all(len(reason) > 10 for reason in rejected["reason"])
        assert "correlates" in rejected.set_index("feature").loc["twin", "reason"]
        assert "below the floor" in rejected.set_index("feature").loc["weak", "reason"]

    def test_log_covers_every_candidate(self, ivs, corr):
        log = scorecard.select_characteristics(ivs, corr)
        assert set(log["feature"]) == set(ivs.index)

    def test_the_cap_is_honoured_and_named(self, ivs, corr):
        log = scorecard.select_characteristics(ivs, corr, max_characteristics=1)
        assert log["selected"].sum() == 1
        assert any("cap of 1" in reason for reason in log["reason"])

    def test_ordering_is_by_information_value(self, ivs, corr):
        log = scorecard.select_characteristics(ivs, corr)
        assert log["iv"].is_monotonic_decreasing
