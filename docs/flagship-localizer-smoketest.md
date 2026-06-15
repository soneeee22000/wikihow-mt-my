# Flagship smoke test — is IFS a useful span-level localizer? (2026-06-15)

The flagship's working thesis was: _IFS fails as a scalar (RQ3) but its components can
localize which spans to repair → selective repair._ This smoke test checks that premise
against the human study's per-item yes/no judgments **before** building any repair pipeline.

Probe: `src/eval/localizer_probe.py` — does each IFS component (quantity/entity/step)
separate the human "yes" from "no" cases? Reported on the _informative_ subset (source
actually contains the feature), since IFS returns 1.0 trivially otherwise.
Result file: `experiments/results/localizer_probe.json`.

## Result — thesis NOT validated in current form

| Component | Informative items  | mean(yes) / mean(no) | AUC  | p    | Verdict                                                                |
| --------- | ------------------ | -------------------- | ---- | ---- | ---------------------------------------------------------------------- |
| Quantity  | 25 (25 yes / 0 no) | 1.0 / —              | n/a  | n/a  | Untestable — numbers reliably copied, no errors to localize            |
| Entity    | 24 (19 / 5)        | 0.66 / 0.40          | 0.65 | 0.30 | Weak directional signal, underpowered. The only live lead.             |
| Step      | 130 (119 / 11)     | 0.90 / 0.84          | 0.55 | 0.33 | No signal (near-degenerate on 1:1 sentence data, as designed)          |
| Action    | —                  | —                    | —    | —    | **No IFS component exists.** Likely the dominant followability driver. |

## Interpretation

The surface-faithfulness axis IFS measures (numbers, entities, step-count) is largely
_orthogonal_ to what makes en→my instructions unfollowable. Followability failures live in
**action/verb fidelity and target-language fluency/meaning**, which a source-anchored surface
metric structurally cannot see. This is consistent with the RQ3 ranking inversion: the
fine-tuned NLLB preserves numbers/entities (high IFS) but mangles actions/fluency (low human
followability).

## Strategic fork for the flagship

- **Option A — reframe (recommended).** Deepen "When Metrics Fail": the claim becomes _the
  source-anchored surface-faithfulness paradigm is mismatched to followability_, evidenced by
  (1) the scalar null, (2) the ranking inversion, (3) this localizer null. Repair, if pursued,
  targets action/meaning guided by a learned/LLM-judge signal — not IFS.
- **Option B — more, harder data.** The entity lead + tiny samples mean a larger, error-denser
  study _might_ rescue component-localization. Higher cost, uncertain payoff.
- **Option C — pivot to "what DOES track followability?"** Build/evaluate a learned (LLM-judge)
  followability metric against the human data; position IFS as the surface baseline it beats.

Recommendation: **A + C combined** (see keystone result below). Do **not** build the naive
IFS-localizer→repair pipeline as the flagship's primary contribution.

## Keystone result — followability IS learnable; IFS just measures the wrong way (2026-06-15)

Point-biserial of each human per-dimension check vs human followability (rating-level, n=320),
with the IFS scalar for contrast:

| Predictor of followability | r          | p       | mean foll (yes / no) |
| -------------------------- | ---------- | ------- | -------------------- |
| human `entities_correct`   | **+0.72**  | 2.6e-52 | 4.50 / 2.48          |
| human `action_correct`     | **+0.68**  | 2.6e-45 | 4.37 / 2.25          |
| human `step_order`         | +0.58      | 1.0e-29 | 4.26 / 2.32          |
| human `quantities_correct` | +0.46      | 6.6e-18 | 4.17 / 2.49          |
| **IFS scalar (automatic)** | **+0.015** | n.s.    | —                    |

The procedural dimensions explain followability strongly (action alone r²≈0.47) — but _only when
measured correctly_. IFS's automatic estimation of those same dimensions captures essentially none
of it. So the failure is not "followability is noise / unpredictable"; the signal is right there.
This crystallizes the flagship thesis:

> **Procedural faithfulness predicts instruction-followability — but surface, source-anchored
> estimation (IFS) of it fails (r≈0) and even inverts system rankings. A learned per-dimension
> estimator (esp. for action, where IFS has no component at all) recovers the signal.**

Complete arc: (1) scalar null + ranking inversion [RQ3] → (2) components don't localize [this
doc] → (3) the dimensions are highly predictive when measured right → build the better estimator.

Next build step (flagship Phase 3): a per-dimension LLM-judge faithfulness estimator (action /
entity / quantity / order), validated against these human labels (AUC/F1), then shown to track
followability where IFS doesn't. Repair, if pursued, is downstream of that estimator.
