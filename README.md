# ADIL

**Subject:** MAIB AI 217 — AI in Finance
**Purpose:** Explainable credit decisioning under UAE supervisory expectations.

MAIB Term 4 · SP Jain School of Global Management, Dubai · Krishna Mathur

## The question

Does the accuracy gained by moving from a weight-of-evidence scorecard to gradient boosting
survive explainability and fairness constraints — and what does it cost per approved
application?

The deliverable is not a model. It is a **constraint ladder**: each rung adds one regulatory
constraint and reports what that constraint cost. A rung that costs nothing is as much a
finding as one that costs a lot.

| Rung | What it adds |
|---|---|
| R0 | Weight-of-evidence scorecard, 20 characteristics — the baseline to beat |
| R1 | LightGBM, unconstrained, 567 features |
| R3 | Monotone constraints on directionally-agreed features |
| R4 | The scorecard's feature budget |
| R5 | Reason-code stability gate (pass/fail; changes no model) |
| R6 | Fairness band on a single approval cutoff |

R2 is absent deliberately. Calibration is applied to every rung and to the baseline alike,
because a miscalibrated model cannot support a cost-based cutoff at all — it is a precondition,
not a constraint the ladder imposes.

## Answer

Gradient boosting genuinely beats the scorecard. **Just over half the accuracy survives the
constraints; less than a fifth of the money does** — 54% of the PR-AUC gain and 19% of the cost
saving per application. The two rates differ because the last rung buys fairness by moving the
approval cutoff, which costs money without costing discrimination. Monotonicity is close to
free; the feature budget is what actually costs.

Four results are negative and are reported as such rather than reframed:

- **Reason codes fail their pre-registered stability gate for every model, the scorecard
  included.** On this evidence they are not fit to serve as adverse-action notices. The
  registration itself turned out to be poorly chosen, which is reported rather than corrected
  after the fact.
- **Excluding sex and age did not make the model fair.** Both remain recoverable from the 20
  features the model uses, at AUC 0.851 and 0.875.
- **Equalising error rates by group did not deliver equalised odds** and cost disparate
  treatment to get part of the way.
- **Reject inference did not work.** Parcelling recovered none of the selection gap at any
  tested k and degraded calibration.

Full numbers in `reports/`, and `docs/index.html` presents them as a single page. Every figure
in both is read from a `metrics/*.json` artifact written by a notebook; none is typed by hand.

## Setup

```bash
uv sync
cp .env.example .env      # DATA_ROOT points at the shared data store
```

Data lives outside the repository and is never committed. See `../data/README.md`.

## Run

```bash
make all
```

`lint` → `test` → `features` → `notebooks` → `card` → `dashboard`. Individually: `make features`
rebuilds the modelling frame, `make notebooks` executes 01–07 headless and clears their outputs,
`make card` regenerates the model card, and `make dashboard` regenerates `docs/index.html`.

Two consecutive runs produce byte-identical metrics and reports. That is checked, not asserted:
DuckDB is pinned to one thread because parallel float aggregation is not associative, and the
frame is explicitly sorted because parallel joins do not promise a row order. Both defects moved
real numbers before they were found.

## Layout

```
src/adil/       features, split, scorecard, evaluation, constraints,
                challenger, reasons, costs, paths
scripts/        build_features.py, render_card.py, build_dashboard.py, install_kernel.py
notebooks/      01_frame … 07_reject_inference
metrics/        every reported number, as JSON
reports/        committed markdown, including model_card.md
docs/           index.html — the generated results dashboard
tests/          mirrors src/adil
```

## Two things that are not obvious

**There is no temporal split, and that is not an oversight.** `application_train` carries no
application date — every `DAYS_*` column is measured relative to the application, so no time
axis exists to order applicants along. `spine.splitting` is deliberately unused and the README
of this project says so where a reader will find it. Manufacturing a pseudo-date would fabricate
a temporal claim the data cannot support. What a temporal split exists to prevent is handled
instead by as-of-application feature construction, enforced in `adil.features` and tested in
`tests/test_features.py` rather than asserted.

**The protected attributes are real, not proxies.** `CODE_GENDER` is sex and `DAYS_BIRTH` is
age; both are genuinely protected under essentially every consumer credit regime, and the
fairness audit is a real audit reported as one. What is absent is nationality and ethnicity —
in a UAE context the axis that matters most — for which Home Credit offers no honest proxy.
That dimension is declared unaddressable rather than approximated.

## Regulatory frame

The governing document is the CBUAE Guidance Note on Consumer Protection and Responsible
Adoption of AI/ML (February 2026). **No clause of it is cited anywhere in this project**, because
the text has not been read by the author of this pipeline. The model card's compliance section
is an explicitly marked stub, and `reports/adil_card.yaml` exposes the evidence as structured
data for MIZAN's BAYAN component, which owns clause mapping for this portfolio.

Nothing here is legal advice.

## Limitations

Public competition data from another market, not UAE consumer data; no figure transfers to an
Emirati lending book without re-estimation. Amounts are in an anonymised, unscaled currency, so
every dirham figure is a labelled scenario rather than a finding. No out-of-time validation is
possible and none is claimed. Full limitations sit in each report and in the model card.
