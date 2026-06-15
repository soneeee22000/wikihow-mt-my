# Flagship Phase 3 — LLM-judge faithfulness probe (cheap validation)

**Date:** 2026-06-15. **Status: thesis SUPPORTED on real human labels.**

**Question.** The floor paper showed the surface IFS metric does not track human
instruction-followability (item-level Pearson r≈0.11; scalar r≈0) and inverts the system
ranking. The flagship thesis is that followability _is_ recoverable — if procedural
faithfulness is **estimated per dimension by a learned judge** rather than read off surface
source-preservation. This probe tests that cheaply, reusing the human labels already collected.

**Method.** `src/eval/llm_judge_probe.py`. A Gemini-2.5-flash judge receives (English source,
Burmese translation) and rates four dimensions yes/no — step order, action/verb, entities,
quantities (temp 0, JSON schema-constrained). Run over all **150** items that have human
labels in `ratings_filled.csv`. Two tests:

- **(A) Recovery** — judge-vs-human agreement on each dimension (accuracy + phi).
- **(B) Usefulness** — correlation of judge scores with human **followability**, compared
  to IFS on the same 150 items.

Results: `experiments/results/llm_judge_probe.json`. Judge labels cached by opaque item id
(no source text) in `experiments/results/llm_judge_cache.json` (gitignored).

## Finding B (headline) — the learned judge recovers the signal IFS misses

| Predictor → human followability          | Pearson r | p      |
| ---------------------------------------- | --------- | ------ |
| **LLM-judge composite (mean of 4 dims)** | **0.54**  | ~0     |
| LLM-judge action_correct                 | 0.44      | ~0     |
| LLM-judge entities_correct               | 0.43      | ~0     |
| LLM-judge step_order                     | 0.34      | 3e-5   |
| LLM-judge quantities_correct             | 0.22      | 0.008  |
| _IFS (item-level, same 150 items)_       | _0.11_    | _0.17_ |
| _IFS (scalar, rating-level)_             | _≈0.0_    | _0.47_ |

The composite judge predicts followability at **r=0.54 vs IFS's 0.11** — same items, same
target. Procedural faithfulness _is_ learnable; IFS just estimates it the wrong way (surface,
source-anchored, saturated).

## Finding A — the judge approximates human per-dimension judgments

| Dimension          | Accuracy | phi      | human yes-rate | judge yes-rate |
| ------------------ | -------- | -------- | -------------- | -------------- |
| action_correct     | 0.89     | **0.49** | 0.85           | 0.95           |
| step_order         | 0.90     | 0.39     | 0.88           | 0.98           |
| entities_correct   | 0.80     | 0.32     | 0.78           | 0.89           |
| quantities_correct | 0.91     | 0.26     | 0.91           | 0.99           |

`action_correct` — the dimension surface IFS has **no automatic component for** — shows the
strongest judge-human correlation. The judge is consistently more lenient than humans (higher
yes-rate everywhere), which caps phi: headroom for a calibrated threshold, an ensemble, or a
stronger/cross-model judge.

## Caveats (honest)

- **Self-preference.** Gemini judges outputs that include Gemini's own translations. It
  predicts a _held-out human_ target (followability), so this mostly threatens Finding A's
  per-dimension agreement, not B's usefulness — but the flagship should confirm with a
  cross-model or held-out judge. (No Anthropic key in `.env`; Gemini is what's wired up.)
- **Single deterministic call** per item; no calibration/ensembling yet — all upside.
- **n=150** real human-validated items; followability IRR is "fair" (α=0.41), so r=0.54 is a
  floor, not a ceiling.

## Implication

The cheap probe clears the bar to build the flagship estimator for real. Natural next steps:
cross-model/held-out judge to kill the self-preference caveat; calibrate the judge (threshold
or regression head) to lift per-dimension recovery; then show the estimator tracks followability
and ranks systems correctly where IFS inverts; repair loop is downstream.
