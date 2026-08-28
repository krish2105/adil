"""ADIL — explainable credit decisioning under UAE supervisory expectations.

MAIB AI 217 (AI in Finance), SP Jain School of Global Management, Dubai.

ADIL asks whether the accuracy gained by moving from a weight-of-evidence
scorecard to gradient boosting survives explainability and fairness constraints,
and what that costs per approved application. The deliverable is a *constraint
ladder*: each rung imposes one further regulatory constraint and reports what it
cost. A rung that costs nothing is as much a finding as one that costs a lot.

Modelling data is Home Credit Default Risk, which is a cross-section: it carries
no application date, so no temporal split is possible and :mod:`spine.splitting`
is deliberately unused. The leakage discipline is instead as-of-application
feature construction, enforced in :mod:`adil.features` and tested rather than
asserted.

The modules:

:mod:`adil.features`
    As-of-application aggregation over the seven Home Credit tables.

:mod:`adil.challenger`
    The gradient boosting challenger and the rungs of the constraint ladder.

:mod:`adil.constraints`
    Monotone credit directions, declared before the challenger is fitted.

:mod:`adil.reasons`
    Adverse-action reason codes and the registered stability gate.

:mod:`adil.scorecard`
    Weight of evidence, information value, and the scorecard baseline.

:mod:`adil.split`
    The stratified split, and why it is not a temporal one.

:mod:`adil.costs`
    The cost model turning a probability into an approval decision.

:mod:`adil.evaluation`
    Discrimination and calibration metrics, and the intervals around them.

:mod:`adil.paths`
    Where the raw tables are, and where artifacts land.
"""

from adil import (
    challenger,
    constraints,
    costs,
    evaluation,
    features,
    paths,
    reasons,
    scorecard,
    split,
)

__version__ = "0.1.0"

__all__ = [
    "challenger",
    "constraints",
    "costs",
    "evaluation",
    "features",
    "paths",
    "reasons",
    "scorecard",
    "split",
]
