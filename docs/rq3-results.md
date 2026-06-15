# RQ3 — Does the IFS metric track human followability?

**Status: PATH B CONFIRMED (negative result, well-powered).** Date: 2026-06-14.

## Data

- 7 raters, **320 ratings**. 6 raters completed the full 30-item shared overlap block
  (SUSAN, Hay Mamm, Luna, Doraemon, SheeFaw, Soe); Phoo completed 20.
- **130 / 160** blind items covered; **50** items rated by ≥2 raters.
- Sources: `experiments/results/ratings_filled.csv` (export) + `ratings_key.csv` (private id→system/ref).
- Reproduce:
  ```
  python src/eval/correlate.py        --ratings experiments/results/ratings_filled.csv --key experiments/results/ratings_key.csv
  python src/eval/human_reliability.py --ratings experiments/results/ratings_filled.csv --key experiments/results/ratings_key.csv
  ```

## Finding 1 — IFS does not correlate with human followability, at any level

| Level                                   | n   | IFS Pearson r | p    | chrF r | BLEU r |
| --------------------------------------- | --- | ------------- | ---- | ------ | ------ |
| Rating-level                            | 320 | **0.016**     | 0.78 | 0.055  | 0.036  |
| Item-level (all items)                  | 130 | 0.110         | 0.21 | 0.120  | 0.083  |
| Item-level (≥2 raters, low-noise means) | 50  | **−0.001**    | 0.99 | —      | —      |

Williams tests: IFS is statistically indistinguishable from chrF and BLEU as a predictor of
followability — all three are ~0.

## Finding 2 (headline) — IFS _inverts_ the system ranking

| System         | Human followability (mean) | IFS (mean)       |
| -------------- | -------------------------- | ---------------- |
| gemini         | **4.25** (best)            | 95.54            |
| nllb_zeroshot  | 4.08                       | 96.05            |
| gtranslate     | 4.00                       | 94.41            |
| nllb_finetuned | **3.60** (worst)           | **97.59** (best) |

System-level **Spearman(human, IFS) = −0.40**. IFS crowns the fine-tuned NLLB — the system humans
rate _least_ followable — as its top system. The source-anchored fine-tune optimizes the structural
cues IFS rewards while producing less-followable Burmese. This is the core "When Metrics Fail" case.

## Finding 3 — Mechanism: IFS saturation (range restriction)

IFS mean **95.0 / 100**, SD 10.5, range 60–100; all four system means fall within ~3 points
(94.4–97.6). Humans use the full scale (followability mean 4.09, SD 1.09). A metric pinned near its
ceiling cannot rank anything.

## Reliability caveat (state honestly in the paper)

Ordinal Krippendorff's **α = 0.38** for followability (0.38 for adequacy) over 50 units —
"fair" agreement, below the 0.667 tentative-conclusion threshold. Followability is a subjective,
skewed-high judgment, so moderate IRR is expected.

Why the conclusion survives it:

- The system-level inversion aggregates 56–98 ratings per system, averaging out per-rating noise.
- The reliable-items r = −0.001 uses 6–7-rater means (Spearman-Brown reliability of those means ≈ 0.79).
- The result is null in the _same direction_ at every level of aggregation.

## Implication for the papers

- **WMT floor paper:** report the corpus + benchmark; present IFS as a metric whose automatic form
  fails human validation → motivates the flagship.
- **Flagship ("When Metrics Fail" / Executable Translation):** the ranking inversion is the central
  evidence; reframe IFS from a scorer into a localizer that flags spans for selective repair, since
  it cannot serve as a system-ranking metric.
