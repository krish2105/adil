# Model card: ADIL — explainable credit decisioning

## Model

- **Name:** ADIL — explainable credit decisioning
- **Version:** 0.1.0 (rung R6, seed 20260827)
- **Owner:** Krishna Mathur, MAIB AI 217, SP Jain School of Global Management, Dubai
- **Model class:** LightGBM, monotone-constrained on 5 of 20 features, capped at 20 features, isotonic-calibrated, single approval cutoff
- **Baseline:** Weight-of-evidence scorecard, 20 characteristics, PDO 20 / 50:1 / 600
- **Status:** Academic coursework. Not deployed and not deployable.

## Intended use

Demonstrating that explanation, fairness evidence and threshold justification can be first-class deliverables in a consumer credit model rather than appendices to one. It answers a research question — whether the accuracy gained by moving from a weight-of-evidence scorecard to gradient boosting survives explainability and fairness constraints, and what it costs per approved application. It is not a lending model. It is fitted on public competition data from another market, has no monitoring, no out-of-time validation and no human review process, and must not be used to decide anything about a real applicant.

## Data

- **Source:** Home Credit Default Risk (Kaggle), all seven tables, 307,511 applications and 574 columns
- **Provenance:** real
- **Not uae data:** Public competition data from another market. The distribution differs materially from any Emirati lending book and no figure here transfers to one without re-estimation.
- **Split:** Stratified 60/20/20 train/calibration/test, seed 20260827, target-rate spread 1.31e-06
- **No temporal split:** application_train carries no application date; every DAYS_* column is relative to the application. No time axis exists, so no temporal split is possible and spine.splitting is deliberately unused. Leakage is controlled instead by as-of-application feature construction, tested in tests/test_features.py.
- **Leakage control:** Every satellite aggregate is filtered to records knowable at application time. Removal counts are recorded for all six tables, including the three where the filter removes nothing.
- **Currency:** Anonymised and unscaled. All costs are reported in dataset currency units. Any dirham figure in this project is a labelled scenario, not a finding.

## Metrics

| Metric | Value | Split |
| --- | --- | --- |
| R0 PR-AUC | 0.21035 | test (stratified holdout, 61,503 applications) |
| R0 AUC | 0.73961 | test (stratified holdout, 61,503 applications) |
| R0 Gini | 0.47921 | test (stratified holdout, 61,503 applications) |
| R0 Brier | 0.06908 | test (stratified holdout, 61,503 applications) |
| R0 Expected calibration error | 0.00353 | test (stratified holdout, 61,503 applications) |
| R0 approval rate | 0.8747 | test, at the rung's cost-optimal cutoff |
| R0 expected cost per approved application | 21,812 dataset currency units | test, at the rung's cost-optimal cutoff |
| R0 reason-code flip rate | 0.762 | test, highest-risk decile, 0.5σ perturbation (gate 0.10: fail) |
| R1 PR-AUC | 0.25869 | test (stratified holdout, 61,503 applications) |
| R1 AUC | 0.77689 | test (stratified holdout, 61,503 applications) |
| R1 Gini | 0.55378 | test (stratified holdout, 61,503 applications) |
| R1 Brier | 0.06671 | test (stratified holdout, 61,503 applications) |
| R1 Expected calibration error | 0.00246 | test (stratified holdout, 61,503 applications) |
| R1 approval rate | 0.8685 | test, at the rung's cost-optimal cutoff |
| R1 expected cost per approved application | 20,378 dataset currency units | test, at the rung's cost-optimal cutoff |
| R1 reason-code flip rate | 0.877 | test, highest-risk decile, 0.5σ perturbation (gate 0.10: fail) |
| R3 PR-AUC | 0.25876 | test (stratified holdout, 61,503 applications) |
| R3 AUC | 0.77556 | test (stratified holdout, 61,503 applications) |
| R3 Gini | 0.55112 | test (stratified holdout, 61,503 applications) |
| R3 Brier | 0.06665 | test (stratified holdout, 61,503 applications) |
| R3 Expected calibration error | 0.00289 | test (stratified holdout, 61,503 applications) |
| R3 approval rate | 0.8508 | test, at the rung's cost-optimal cutoff |
| R3 expected cost per approved application | 20,836 dataset currency units | test, at the rung's cost-optimal cutoff |
| R3 reason-code flip rate | 0.854 | test, highest-risk decile, 0.5σ perturbation (gate 0.10: fail) |
| R4 PR-AUC | 0.23656 | test (stratified holdout, 61,503 applications) |
| R4 AUC | 0.76186 | test (stratified holdout, 61,503 applications) |
| R4 Gini | 0.52371 | test (stratified holdout, 61,503 applications) |
| R4 Brier | 0.06772 | test (stratified holdout, 61,503 applications) |
| R4 Expected calibration error | 0.00245 | test (stratified holdout, 61,503 applications) |
| R4 approval rate | 0.8927 | test, at the rung's cost-optimal cutoff |
| R4 expected cost per approved application | 20,496 dataset currency units | test, at the rung's cost-optimal cutoff |
| R4 reason-code flip rate | 0.75 | test, highest-risk decile, 0.5σ perturbation (gate 0.10: fail) |
| R6 PR-AUC | 0.23656 | test (stratified holdout, 61,503 applications) |
| R6 AUC | 0.76186 | test (stratified holdout, 61,503 applications) |
| R6 Gini | 0.52371 | test (stratified holdout, 61,503 applications) |
| R6 Brier | 0.06772 | test (stratified holdout, 61,503 applications) |
| R6 Expected calibration error | 0.00245 | test (stratified holdout, 61,503 applications) |
| R6 approval rate | 0.9355 | test, at the rung's cost-optimal cutoff |
| R6 expected cost per approved application | 20,118 dataset currency units | test, at the rung's cost-optimal cutoff |
| R6 reason-code flip rate | 0.75 | test, highest-risk decile, 0.5σ perturbation (gate 0.10: fail) |

## Fairness

- **Attribute:** Sex (CODE_GENDER), audited alongside age band (DAYS_BIRTH)
- **Attribute status:** protected
- **Note:** These are genuine protected attributes, not proxies. The project brief assumed otherwise and the brief was wrong. Neither attribute is a model input — deciding on them would be disparate treatment — and both are retained for measurement only.
- **Unaddressable:** Nationality and ethnicity are absent from Home Credit and no honest proxy exists. In a UAE context that is the axis that matters most. It is declared unaddressable rather than approximated, and nothing in this project speaks to it.
- **Prioritised metric:** Calibration by group. A group whose predicted probabilities exceed its observed default rate is charged for risk it does not carry, and no downstream threshold repairs that. The equalised-odds gap is reported in full rather than optimised away.
- **Approval disparity sex:** 0.9555
- **Approval disparity age:** 0.8207
- **Disparity floor imposed:** 0.8
- **Single cutoff:** One cutoff for every applicant. Equalising error rates by group was shown to require a different cutoff per sex, which is disparate treatment, and it closed the true-positive gap without delivering equalised odds.
- **Proxy audit:** Excluding the attributes prevents disparate treatment, not disparate impact. From the 20 features the model actually uses, sex is recoverable at AUC 0.851 and age band at 0.875. Removing the columns did not remove the information.

## Limitations

- Public competition data from another market, not UAE consumer data.
- No time axis exists in the source, so no out-of-time validation is possible and none is claimed. Metric stability over time is untested and would be a monitoring requirement before any deployment.
- Reason-code stability fails its pre-registered gate for every model tested, the scorecard included: 75.0% of declined applicants see their top-3 reasons change under a 0.5σ perturbation, against a registered ceiling of 10%. The gate was registered before measurement and has not been revised. On this evidence the reason codes are not fit to serve as adverse-action notices.
- The registered perturbation scale turned out not to discriminate between model classes, since no model passes at it. That is a fault in the registration, reported rather than corrected after the fact.
- The cost ratio is published (UCI German Credit, 5:1); the loss given default, margin rate and AED exposure are assumed. No dirham figure is a measurement.
- A single cost matrix treats every applicant as carrying the same loss and margin. Exposures on this book span a factor of 70; the exposure-weighted sensitivity is reported in reports/decision_table.md.
- Reject inference did not work. Parcelling recovered none of the selection gap at any tested k, degrading PR-AUC by up to 0.0126 and calibration further. Nothing else in this project depends on it.
- Hyperparameters are fixed across rungs and untuned. The ladder measures what constraints cost, not how high the AUC can be pushed.
- Intervals cover measurement on one test split. They do not cover how much a metric would move on different training data, which is larger.
- This card is an academic artifact. It is the author's reading of published guidance, not legal advice, and it has not been reviewed by a lawyer or a compliance function.

## Regulatory mapping — PENDING

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
| Governance and accountability | Named owner, versioned model, recorded seed, reproducible pipeline | `metrics/*.json` |
| Fairness and non-discrimination | Group rates, disparity ratios, calibration by group, proxy audit | `reports/fairness_report.md` |
| Transparency and explainability | Adverse-action reason codes and a pre-registered stability gate | `reports/explainability.md` |
| Data quality | As-of-application filters with row counts, feature manifest, schema validation | `reports/data_quality.md` |
| Human oversight | Cost matrix and threshold justification, one cutoff for every applicant | `reports/decision_table.md` |
| Model risk and challenger review | Constraint ladder, baseline versus challenger, reject-inference sensitivity | `reports/challenger.md` |
| Continuous monitoring | **Not produced.** No out-of-time validation is possible in this data | — |
| Third-party risk | **Not applicable.** No third-party model or data provider is used | — |

Nothing in this table is a compliance claim. It is an index of what exists.

## Provenance

Every number in this card is read from a `metrics/*.json` artifact written by a notebook. None is typed by hand. Deleting `metrics/` and rerunning `make all` reproduces this file exactly.

Declared monotone directions: 9 rules in `adil.constraints`, committed before the challenger was fitted.

Reason-code gate: registered in `adil.reasons` before any flip rate was measured.

Cost ratio: UCI Statlog German Credit Data, german.doc section 8 'Cost Matrix': 'It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).' Reproduced as a ratio only — German Credit's amounts are in Deutsche Marks and do not transfer to Home Credit.

Scorecard selection: IV floor 0.02, maximum correlation 0.7, cap 20 characteristics.
