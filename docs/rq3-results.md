# RQ3 — Does the IFS metric track human followability?

**Status: PATH B CONFIRMED (negative result, well-powered).** Date: 2026-07-11
(updated from the 2026-06-15 write-up at 8 raters / 370 ratings).

## Data

- **9 raters, 420 ratings.** Eight completed the full 50-item set (cherry, SUSAN, Hay Mamm,
  Luna, Doraemon, SheeFaw, Soe, Thidar); Phoo contributed 20.
- **160 / 160** blind items covered. **60** items carry ≥2 ratings (the units for Krippendorff's α);
  the 30-item shared overlap block carries up to 9 ratings each.
- Sources: `experiments/results/ratings_filled.csv` (export) + `ratings_key.csv` (private id→system/ref).
- Reproduce:
  ```
  python src/eval/correlate.py         --ratings experiments/results/ratings_filled.csv --key experiments/results/ratings_key.csv
  python src/eval/human_reliability.py --ratings experiments/results/ratings_filled.csv --key experiments/results/ratings_key.csv
  python src/eval/llm_judge_probe.py   --ratings experiments/results/ratings_filled.csv
  python src/eval/estimator_ranking.py --ratings experiments/results/ratings_filled.csv --key experiments/results/ratings_key.csv
  python src/eval/make_tables.py
  ```

## Aggregation (this matters — it changed the headline)

Per-system means are computed over the **40 items per system** (each item once), **not over
ratings**. The overlap block is unbalanced across systems (9 gtranslate, 9 nllb_finetuned,
8 nllb_zeroshot, but only 4 gemini) and carries up to 9 ratings per item, so a rating-weighted
mean over-weights it and distorts the system ordering.

The earlier write-up used rating-weighted means. Under that (incorrect) weighting the fine-tuned
NLLB appeared _last_ on human followability and system Spearman(human, IFS) came out at −0.80,
which is where the retired "IFS **inverts** the ranking" claim came from. Under balanced
item-level aggregation the fine-tuned NLLB is **3rd of 4** and ρ = **+0.20**. `human_reliability.py`
and `estimator_ranking.py` now agree exactly.

## Finding 1 — IFS does not correlate with human followability, at any level

| Level        | n   | IFS Pearson r | p    | chrF r | BLEU r |
| ------------ | --- | ------------- | ---- | ------ | ------ |
| Rating-level | 420 | **0.084**     | 0.09 | 0.099  | 0.033  |
| Item-level   | 160 | **0.128**     | 0.11 | 0.202  | 0.060  |

Every CI straddles zero. Williams tests: IFS is statistically indistinguishable from chrF and
BLEU as a predictor of followability (p = 0.81, 0.49) — all three are ~0.

## Finding 2 (headline) — IFS _cannot rank_ systems

| System         | Human followability (mean, 40 items) | IFS (mean)       |
| -------------- | ------------------------------------ | ---------------- |
| gemini         | **4.37** (best)                      | 95.00 (3rd)      |
| gtranslate     | 4.24                                 | 95.33            |
| nllb_finetuned | 3.78 (3rd)                           | **96.71** (best) |
| nllb_zeroshot  | **3.64** (worst)                     | 94.46            |

- IFS spans **2.25 points** (94.46–96.71) across four systems humans separate from 3.64 to 4.37.
- Descriptive system-level Spearman(human, IFS) = **+0.20** (n=4 — never reported with a p-value).
- IFS awards its **highest** score to the fine-tuned NLLB, which humans place **3rd of 4**, and
  ranks **Gemini — humans' clear best — only 3rd**.

The claim is **"IFS cannot rank"**, not "IFS inverts". With four system means inside a 2.25-point
band, the n=4 Spearman is decided by sub-point noise and flips sign under changes in aggregation
or panel composition. That instability _is_ the range-restriction finding — but no conclusion may
rest on the sign. Do not reintroduce an "inversion" claim.

## Finding 3 — Mechanism: IFS saturation (range restriction)

IFS mean **95.4 / 100**, sd 9.9, range 60–100; all four system means within 2.25 points. Humans use
the full scale (followability mean 4.01, sd 1.19). A metric pinned near its ceiling cannot rank.

## Finding 4 (constructive) — a learned per-dimension judge recovers what IFS loses

Same four dimensions (step, action, entity, quantity), estimated by an LLM judge instead of by
surface preservation. Cached labels in `llm_judge_cache.json` (160 items).

| Predictor                 | Item-level Pearson r | 95% CI        | Williams p vs IFS | System ρ |
| ------------------------- | -------------------- | ------------- | ----------------- | -------- |
| IFS (source-anchored)     | 0.128                | [−0.04, 0.29] | —                 | 0.20     |
| Learned estimator         | **0.531**            | [0.37, 0.66]  | <0.001            | **1.00** |
| ↳ Gemini held out (n=120) | **0.549**            | [0.39, 0.68]  | <0.001            | **1.00** |

Per-dimension vs followability: entity 0.444, action 0.407, step 0.313, quantity 0.202.
The judge is more lenient than humans on every dimension (it accepts all of Gemini's own items),
so absolute scores over-credit fluent output — but the recovery survives holding Gemini out, so it
is not self-preference.

## Reliability caveat (state honestly in the paper)

Ordinal Krippendorff's **α = 0.39** for followability (0.40 for adequacy) over the 60 multiply-rated
items — "fair" agreement, below the 0.667 tentative-conclusion threshold. Followability is a
subjective, skewed-high judgment, so moderate IRR is expected.

Why the conclusion survives it:

- The result is null in the _same direction_ at every level of aggregation (rating, item, system).
- Each per-system mean pools 40 items (75–121 ratings), so per-rating disagreement cancels.
- The saturation argument is about the _width of the IFS band_, which does not depend on rater noise.

## Implication for the papers

- **WMT floor paper:** report the corpus + benchmark; present IFS as a metric whose automatic form
  fails human validation → motivates the flagship.
- **Flagship ("When Metrics Fail" / Executable Translation):** the ranking _failure_ plus the
  learned-estimator recovery is the central evidence; reframe IFS from a scorer into a localizer
  that flags spans for selective repair, since it cannot serve as a system-ranking metric.
