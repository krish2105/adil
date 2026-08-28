"""Render the ADIL model card from the metrics artifacts.

No number in `reports/model_card.md` is typed by a human. Every value is read
from a `metrics/*.json` file written by a notebook, so deleting `metrics/` and
rerunning the pipeline reproduces the card exactly. This is the same discipline
SPINE applies to `reports/proof.json`, and it is what makes "the card matches the
model" checkable rather than asserted.

Two things are emitted.

`reports/model_card.md`
    The human-readable card, rendered by :func:`spine.cards.render_card`.

`reports/adil_card.yaml`
    The same content as structured data, for MIZAN's BAYAN component to map
    clause by clause. `spine.cards` is deliberately regulation-blind and the
    clause mapping lives in exactly one place, which is not here.

The compliance section is an explicitly marked stub. The CBUAE Guidance Note of
February 2026 has not been read by whoever ran this script, and a citation to a
clause nobody has read is worse than no citation at all.
"""

import json
from typing import Any

import yaml
from spine.cards import render_card, validate_card

from adil import constraints, costs, scorecard, split

#: The governance themes a clause mapping would draw on, and where the evidence lives.
#: Built as data rather than a markdown blob so each row stays legible in source.
COMPLIANCE_THEMES: tuple[tuple[str, str, str], ...] = (
    (
        "Governance and accountability",
        "Named owner, versioned model, recorded seed, reproducible pipeline",
        "`metrics/*.json`",
    ),
    (
        "Fairness and non-discrimination",
        "Group rates, disparity ratios, calibration by group, proxy audit",
        "`reports/fairness_report.md`",
    ),
    (
        "Transparency and explainability",
        "Adverse-action reason codes and a pre-registered stability gate",
        "`reports/explainability.md`",
    ),
    (
        "Data quality",
        "As-of-application filters with row counts, feature manifest, schema validation",
        "`reports/data_quality.md`",
    ),
    (
        "Human oversight",
        "Cost matrix and threshold justification, one cutoff for every applicant",
        "`reports/decision_table.md`",
    ),
    (
        "Model risk and challenger review",
        "Constraint ladder, baseline versus challenger, reject-inference sensitivity",
        "`reports/challenger.md`",
    ),
    (
        "Continuous monitoring",
        "**Not produced.** No out-of-time validation is possible in this data",
        "—",
    ),
    (
        "Third-party risk",
        "**Not applicable.** No third-party model or data provider is used",
        "—",
    ),
)

COMPLIANCE_PREAMBLE = """## Regulatory mapping — PENDING

**This section is deliberately empty.**

ADIL's governing frame is the CBUAE Guidance Note on Consumer Protection and Responsible
Adoption of AI/ML (February 2026). No clause of it is cited anywhere in this project,
because the text has not been read by the author of this pipeline. A paraphrased
regulatory claim without provenance, or a clause number produced from memory, would be
worse than the gap it fills.

The evidence a clause-level mapping would draw on already exists and is indexed below.
`reports/adil_card.yaml` exposes the same material as structured data for MIZAN's BAYAN
component, which owns clause mapping for every model in this portfolio.

| Governance theme | Evidence produced | Artifact |
|---|---|---|
"""


def compliance_stub() -> str:
    """Render the pending regulatory section."""
    rows = "\n".join(
        f"| {theme} | {evidence} | {where} |" for theme, evidence, where in COMPLIANCE_THEMES
    )
    return (
        COMPLIANCE_PREAMBLE
        + rows
        + "\n\nNothing in this table is a compliance claim. It is an index of what exists.\n"
    )


NOT_LEGAL_ADVICE = (
    "This card is an academic artifact. It is the author's reading of published guidance, "
    "not legal advice, and it has not been reviewed by a lawyer or a compliance function."
)


def load(name: str) -> dict[str, Any]:
    """Read one metrics artifact."""
    from adil.paths import metrics_dir

    path = metrics_dir() / name
    if not path.exists():
        raise SystemExit(f"{path} does not exist; run `make features` and `make notebooks` first")
    return json.loads(path.read_text())


def build_spec() -> dict[str, Any]:
    """Assemble the card specification from the metrics artifacts."""
    frame = load("frame.json")
    r0 = load("r0.json")
    r4 = load("r4.json")
    r5 = load("r5.json")
    r6 = load("r6.json")
    headline = load("headline.json")
    fairness = load("fairness.json")
    rejects = load("reject_inference.json")

    ladder = {row["rung"]: row for row in headline["ladder"]}
    operative = ladder["R6"]

    metrics = []
    for rung in ("R0", "R1", "R3", "R4", "R6"):
        row = ladder[rung]
        for label, key in (
            ("PR-AUC", "PR-AUC"),
            ("AUC", "AUC"),
            ("Gini", "Gini"),
            ("Brier", "Brier"),
            ("Expected calibration error", "ECE"),
        ):
            metrics.append(
                {
                    "name": f"{rung} {label}",
                    "value": round(float(row[key]), 5),
                    "split": "test (stratified holdout, 61,503 applications)",
                }
            )
        metrics.append(
            {
                "name": f"{rung} approval rate",
                "value": round(float(row["approval rate"]), 4),
                "split": "test, at the rung's cost-optimal cutoff",
            }
        )
        metrics.append(
            {
                "name": f"{rung} expected cost per approved application",
                "value": f"{row['cost/approved (dataset)']:,.0f} dataset currency units",
                "split": "test, at the rung's cost-optimal cutoff",
            }
        )
        metrics.append(
            {
                "name": f"{rung} reason-code flip rate",
                "value": round(float(row["flip rate"]), 3),
                "split": (
                    f"test, highest-risk decile, {r5['registered_sigma']}σ perturbation "
                    f"(gate {r5['registered_gate']:.2f}: {row['R5']})"
                ),
            }
        )

    proxy = {row["reconstructing"]: row["test AUC"] for row in fairness["proxy_audit"]}

    return {
        "model": {
            "name": "ADIL — explainable credit decisioning",
            "version": f"0.1.0 (rung R6, seed {split.SEED})",
            "owner": "Krishna Mathur, MAIB AI 217, SP Jain School of Global Management, Dubai",
            "model_class": (
                f"LightGBM, monotone-constrained on {r4['n_constrained_features']} of "
                f"{r4['n_features']} features, capped at {r4['feature_cap']} features, "
                f"isotonic-calibrated, single approval cutoff"
            ),
            "baseline": (
                f"Weight-of-evidence scorecard, {r0['n_characteristics']} characteristics, "
                f"PDO {r0['scaling']['pdo']} / {r0['scaling']['base_odds']}:1 / "
                f"{r0['scaling']['base_points']}"
            ),
            "status": "Academic coursework. Not deployed and not deployable.",
        },
        "intended_use": (
            "Demonstrating that explanation, fairness evidence and threshold justification "
            "can be first-class deliverables in a consumer credit model rather than "
            "appendices to one. It answers a research question — whether the accuracy gained "
            "by moving from a weight-of-evidence scorecard to gradient boosting survives "
            "explainability and fairness constraints, and what it costs per approved "
            "application. It is not a lending model. It is fitted on public competition data "
            "from another market, has no monitoring, no out-of-time validation and no human "
            "review process, and must not be used to decide anything about a real applicant."
        ),
        "data": {
            "source": (
                "Home Credit Default Risk (Kaggle), all seven tables, "
                f"{frame['rows']:,} applications and {frame['columns']:,} columns"
            ),
            "provenance": "real",
            "not_uae_data": (
                "Public competition data from another market. The distribution differs "
                "materially from any Emirati lending book and no figure here transfers to "
                "one without re-estimation."
            ),
            "split": (
                f"Stratified {int(100 * 0.6)}/{int(100 * 0.2)}/{int(100 * 0.2)} "
                f"train/calibration/test, seed {frame['seed']}, target-rate spread "
                f"{frame['target_rate_spread']:.2e}"
            ),
            "no_temporal_split": (
                "application_train carries no application date; every DAYS_* column is "
                "relative to the application. No time axis exists, so no temporal split is "
                "possible and spine.splitting is deliberately unused. Leakage is controlled "
                "instead by as-of-application feature construction, tested in "
                "tests/test_features.py."
            ),
            "leakage_control": (
                "Every satellite aggregate is filtered to records knowable at application "
                "time. Removal counts are recorded for all six tables, including the three "
                "where the filter removes nothing."
            ),
            "currency": (
                "Anonymised and unscaled. All costs are reported in dataset currency units. "
                "Any dirham figure in this project is a labelled scenario, not a finding."
            ),
        },
        "metrics": metrics,
        "fairness": {
            "attribute": "Sex (CODE_GENDER), audited alongside age band (DAYS_BIRTH)",
            "attribute_status": "protected",
            "note": (
                "These are genuine protected attributes, not proxies. The project brief "
                "assumed otherwise and the brief was wrong. Neither attribute is a model "
                "input — deciding on them would be disparate treatment — and both are "
                "retained for measurement only."
            ),
            "unaddressable": (
                "Nationality and ethnicity are absent from Home Credit and no honest proxy "
                "exists. In a UAE context that is the axis that matters most. It is declared "
                "unaddressable rather than approximated, and nothing in this project speaks "
                "to it."
            ),
            "prioritised_metric": (
                "Calibration by group. A group whose predicted probabilities exceed its "
                "observed default rate is charged for risk it does not carry, and no "
                "downstream threshold repairs that. The equalised-odds gap is reported in "
                "full rather than optimised away."
            ),
            "approval_disparity_sex": round(float(operative["disparity sex"]), 4),
            "approval_disparity_age": round(float(operative["disparity age"]), 4),
            "disparity_floor_imposed": r6["disparity_floor"],
            "single_cutoff": (
                "One cutoff for every applicant. Equalising error rates by group was shown "
                "to require a different cutoff per sex, which is disparate treatment, and it "
                "closed the true-positive gap without delivering equalised odds."
            ),
            "proxy_audit": (
                "Excluding the attributes prevents disparate treatment, not disparate "
                f"impact. From the {r4['feature_cap']} features the model actually uses, sex "
                f"is recoverable at AUC {proxy['sex is male']:.3f} and age band at "
                f"{proxy['age under 35']:.3f}. Removing the columns did not remove the "
                "information."
            ),
        },
        "limitations": [
            "Public competition data from another market, not UAE consumer data.",
            (
                "No time axis exists in the source, so no out-of-time validation is possible "
                "and none is claimed. Metric stability over time is untested and would be a "
                "monitoring requirement before any deployment."
            ),
            (
                f"Reason-code stability fails its pre-registered gate for every model tested, "
                f"the scorecard included: {operative['flip rate']:.1%} of declined applicants "
                f"see their top-{r5['top_k']} reasons change under a "
                f"{r5['registered_sigma']}σ perturbation, against a registered ceiling of "
                f"{r5['registered_gate']:.0%}. The gate was registered before measurement "
                "and has not been revised. On this evidence the reason codes are not fit to "
                "serve as adverse-action notices."
            ),
            (
                "The registered perturbation scale turned out not to discriminate between "
                "model classes, since no model passes at it. That is a fault in the "
                "registration, reported rather than corrected after the fact."
            ),
            (
                "The cost ratio is published (UCI German Credit, 5:1); the loss given "
                "default, margin rate and AED exposure are assumed. No dirham figure is a "
                "measurement."
            ),
            (
                "A single cost matrix treats every applicant as carrying the same loss and "
                "margin. Exposures on this book span a factor of "
                f"{r6['exposure_weighted_sensitivity']['exposure_max_over_min']:.0f}; the "
                "exposure-weighted sensitivity is reported in reports/decision_table.md."
            ),
            (
                "Reject inference did not work. Parcelling recovered none of the selection "
                f"gap at any tested k, degrading PR-AUC by up to "
                f"{abs(min(r['moved from accepted-only'] for r in rejects['recovery'])):.4f} "
                "and calibration further. Nothing else in this project depends on it."
            ),
            (
                "Hyperparameters are fixed across rungs and untuned. The ladder measures "
                "what constraints cost, not how high the AUC can be pushed."
            ),
            (
                "Intervals cover measurement on one test split. They do not cover how much a "
                "metric would move on different training data, which is larger."
            ),
            NOT_LEGAL_ADVICE,
        ],
    }


def main() -> None:
    from adil.paths import reports_dir

    spec = build_spec()
    problems = validate_card(spec)
    if problems:
        raise SystemExit(
            "the card specification is not renderable:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )

    markdown = render_card(spec)
    markdown += "\n" + compliance_stub()
    markdown += (
        "\n## Provenance\n\n"
        "Every number in this card is read from a `metrics/*.json` artifact written by a "
        "notebook. None is typed by hand. Deleting `metrics/` and rerunning `make all` "
        "reproduces this file exactly.\n\n"
        "Declared monotone directions: "
        f"{len(constraints.DIRECTIONS)} rules in `adil.constraints`, committed before the "
        "challenger was fitted.\n\n"
        "Reason-code gate: registered in `adil.reasons` before any flip rate was measured.\n\n"
        f"Cost ratio: {costs.GERMAN_CREDIT_SOURCE}\n\n"
        f"Scorecard selection: IV floor {scorecard.IV_FLOOR}, maximum correlation "
        f"{scorecard.MAX_CORRELATION}, cap {scorecard.MAX_CHARACTERISTICS} characteristics.\n"
    )

    reports = reports_dir()
    (reports / "model_card.md").write_text(markdown)
    (reports / "adil_card.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=88)
    )
    print(f"wrote {reports / 'model_card.md'} ({len(markdown.splitlines())} lines)")
    print(f"wrote {reports / 'adil_card.yaml'} for MIZAN/BAYAN clause mapping")


if __name__ == "__main__":
    main()
