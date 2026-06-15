# RQ3 — Does the IFS metric track human followability?

**Status: PATH B CONFIRMED (negative result, well-powered).** Date: 2026-06-15 (updated; was 2026-06-14 at 7 raters/320).

## Data

- 8 raters, **370 ratings**. 7 raters completed the full 50-item set
  (cherry, SUSAN, Hay Mamm, Luna, Doraemon, SheeFaw, Soe); Phoo completed 20.
- **150 / 160** blind items covered; **50** items rated by ≥2 raters.
- The 8th rater (cherry, +50) strengthened every result: α rose 0.38→0.41, coverage 130→150 items,
  and the system-level ranking inversion sharpened from Spearman −0.40 to −0.80.
- Sources: `experiments/results/ratings_filled.csv` (export) + `ratings_key.csv` (private id→system/ref).
- Reproduce:
  ```
  python src/eval/correlate.py        --ratings experiments/results/ratings_filled.csv --key experiments/results/ratings_key.csv
  python src/eval/human_reliability.py --ratings experiments/results/ratings_filled.csv --key experiments/results/ratings_key.csv
  ```

## Finding 1 — IFS does not correlate with human followability, at any level

| Level                  | n   | IFS Pearson r | p    | chrF r | BLEU r |
| ---------------------- | --- | ------------- | ---- | ------ | ------ |
| Rating-level           | 370 | **0.038**     | 0.47 | 0.083  | 0.034  |
| Item-level (all items) | 150 | 0.111         | 0.17 | 0.181  | 0.077  |

Williams tests: IFS is statistically indistinguishable from chrF and BLEU as a predictor of
followability — all three are ~0.

## Finding 2 (headline) — IFS _inverts_ the system ranking

| System         | Human followability (mean) | IFS (mean)       |
| -------------- | -------------------------- | ---------------- |
| gemini         | **4.35** (best)            | 95.76            |
| gtranslate     | 4.06                       | 94.35            |
| nllb_zeroshot  | 3.96                       | 96.32            |
| nllb_finetuned | **3.58** (worst)           | **97.56** (best) |

System-level **Spearman(human, IFS) = −0.80** (descriptive, n=4 — never reported with a p-value).
IFS crowns the fine-tuned NLLB — the system humans
rate _least_ followable — as its top system. The source-anchored fine-tune optimizes the structural
cues IFS rewards while producing less-followable Burmese. This is the core "When Metrics Fail" case.

## Finding 3 — Mechanism: IFS saturation (range restriction)

IFS mean **95.0 / 100**, SD 10.5, range 60–100; all four system means fall within ~3 points
(94.4–97.6). Humans use the full scale (followability mean 4.09, SD 1.09). A metric pinned near its
ceiling cannot rank anything.

## Reliability caveat (state honestly in the paper)

Ordinal Krippendorff's **α = 0.41** for followability (0.40 for adequacy) over 50 units —
"fair" agreement, below the 0.667 tentative-conclusion threshold. Followability is a subjective,
skewed-high judgment, so moderate IRR is expected. (Rose from 0.38 at 7 raters after cherry's full set.)

Why the conclusion survives it:

- The system-level inversion aggregates 66–110 ratings per system, averaging out per-rating noise.
- The overlap-block items are now rated by up to 7 raters, so their means are low-noise.
- The result is null in the _same direction_ at every level of aggregation.

## Implication for the papers

- **WMT floor paper:** report the corpus + benchmark; present IFS as a metric whose automatic form
  fails human validation → motivates the flagship.
- **Flagship ("When Metrics Fail" / Executable Translation):** the ranking inversion is the central
  evidence; reframe IFS from a scorer into a localizer that flags spans for selective repair, since
  it cannot serve as a system-ranking metric.
