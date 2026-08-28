"""Monotone constraints are a declaration, and a declaration has obligations.

Rung R3 constrains the challenger to be monotone in features whose credit
direction is agreed in advance. The value of the exercise depends entirely on
"in advance" being true, so these tests check the properties that keep the
declaration honest rather than decorative: every constraint carries a reason,
nothing is constrained by default, and the protected attributes can never appear.
"""

import pytest

from adil import constraints, scorecard


class TestDeclaration:
    def test_every_direction_states_a_reason(self):
        for direction in constraints.DIRECTIONS:
            assert direction.sign in (-1, 1)
            assert len(direction.rationale) > 30, f"{direction.pattern} has no real rationale"

    def test_patterns_are_anchored_or_specific(self):
        # An unanchored pattern that happens to match a future feature name would
        # constrain it silently. Every pattern must be deliberate.
        for direction in constraints.DIRECTIONS:
            assert direction.pattern.startswith("^") or "_" in direction.pattern

    def test_no_protected_attribute_is_constrained(self):
        # Constraining a protected attribute would mean it had entered the model.
        assigned = constraints.constraint_table(list(scorecard.PROTECTED_ATTRIBUTES))
        assert (assigned["constraint"] == 0).all()


class TestAssignment:
    def test_unlisted_features_are_left_free(self):
        # The default is zero. A feature is constrained because someone decided it
        # should be, never because a rule swept it up.
        assert constraints.monotone_constraints(["SOME_UNDECLARED_AGGREGATE"]) == [0]

    def test_external_scores_reduce_predicted_risk(self):
        assert constraints.monotone_constraints(["EXT_SOURCE_1", "EXT_SOURCE_2"]) == [-1, -1]

    def test_delinquency_increases_predicted_risk(self):
        assigned = constraints.monotone_constraints(
            ["BUR_CREDIT_DAY_OVERDUE_MAX", "POS_SK_DPD_MEAN", "BUR_CNT_CREDIT_PROLONG_SUM"]
        )
        assert assigned == [1, 1, 1]

    def test_delinquent_bureau_statuses_are_constrained_but_current_is_not(self):
        # STATUS 1-5 are months of arrears. STATUS 0 is a month paid on time, whose
        # direction is not obvious: a high share of on-time months can simply mean a
        # long open loan. Constraining it would be inventing a direction.
        assert constraints.monotone_constraints(["BB_STATUS_3_SHARE"]) == [1]
        assert constraints.monotone_constraints(["BB_STATUS_0_SHARE"]) == [0]

    def test_employment_tenure_direction_accounts_for_the_sign_of_days(self):
        # DAYS_EMPLOYED is negative and counts back from the application, so a value
        # closer to zero is a shorter tenure. Risk therefore increases with it.
        assert constraints.monotone_constraints(["DAYS_EMPLOYED"]) == [1]

    def test_order_follows_the_feature_list(self):
        assigned = constraints.monotone_constraints(
            ["SOMETHING_FREE", "EXT_SOURCE_1", "POS_SK_DPD_MAX"]
        )
        assert assigned == [0, -1, 1]

    def test_length_always_matches_the_feature_list(self):
        features = ["EXT_SOURCE_1", "A", "B", "POS_SK_DPD_MAX", "C"]
        assert len(constraints.monotone_constraints(features)) == len(features)


class TestConstraintTable:
    def test_names_the_matched_rule_for_every_constrained_feature(self):
        table = constraints.constraint_table(["EXT_SOURCE_1", "UNDECLARED"])
        by_feature = table.set_index("feature")
        assert by_feature.loc["EXT_SOURCE_1", "constraint"] == -1
        assert len(by_feature.loc["EXT_SOURCE_1", "rationale"]) > 30
        assert by_feature.loc["UNDECLARED", "constraint"] == 0
        assert by_feature.loc["UNDECLARED", "rationale"] == "no agreed credit direction"

    def test_covers_every_feature_given(self):
        features = ["EXT_SOURCE_1", "UNDECLARED", "POS_SK_DPD_MAX"]
        assert constraints.constraint_table(features)["feature"].tolist() == features

    def test_a_feature_matching_two_rules_is_an_error(self):
        # Two rules matching one feature means the declaration disagrees with itself,
        # and silently taking the first would hide that.
        with pytest.raises(ValueError, match="matches more than one"):
            constraints.constraint_table(
                ["EXT_SOURCE_1"],
                directions=(
                    constraints.Direction("^EXT_SOURCE_1$", -1, "a" * 40),
                    constraints.Direction("EXT_SOURCE", 1, "b" * 40),
                ),
            )
