# Phase 1 Data Audit — WMT2026 WikiHow en→my

**Date:** 2026-06-06 · **Method:** read-only pass (`_phase1_audit.py`) over every candidate CSV + the split-creation notebooks. Raw numbers in `_phase1_audit_report.json`.

## TL;DR

- **Canonical corpus = `Data10KFinal.csv` (10,094 rows, ≥86 WikiHow articles).** Use this; archive the rest.
- **Existing `train/test/validate.csv` splits are INVALID** (sentence-level random, unseeded → article leakage + exact-pair leakage). **Must rebuild article-disjoint, seeded.**
- **All baseline evals in `lab reports/eva_*.ipynb` were computed on the leaky split → must be re-run after rebuild.**
- Needs: dedup (38 exact-dup rows), normalization (numerals, whitespace), and an `article_id` (does not yet exist).

## 1. Provenance (matches owner's account: MTPE pipeline)

| Component file                                            | Rows       | Articles (≈) | Role                                          |
| --------------------------------------------------------- | ---------- | ------------ | --------------------------------------------- |
| `human_written_data_5K.csv` = `en_mm_parallel_corpus.csv` | 5,021      | 44           | Professionally translated (BharSarPyan), QC'd |
| `6Kdata.csv` / `6Kmyen.csv`                               | 6,426      | 54           | Volunteer MTPE (circle post-editing of MT)    |
| **`Data10KFinal.csv`**                                    | **10,094** | **≥86**      | 5K + 6K merged, deduplicated → canonical      |
| `enmy10K.csv` (mislabeled)                                | 3,670      | 32           | Older partial; NOT the 10k. Ignore.           |
| `master_spreadsheet.csv` = `athousandmore.csv`            | 1,405      | 10           | Per-article MTPE staging chunk                |

**Datasheet phrasing (honest):** "~5,000 pairs were professionally translated and ~6,400 were produced by volunteer machine-translation post-editing (MTPE); the merged, deduplicated corpus of 10,094 pairs was quality-checked in full by a professional Myanmar translation service." Never call this "gold-standard human translation."

## 2. Split integrity — FAIL

Split logic (`recombine10K.ipynb`): `train_test_split(data, test_size=0.3)` then split again — **no `random_state`, no grouping by article.**

Measured exact (en|||my)-pair overlap across the saved splits:
| Split set | Sizes (train/test/valid) | train∩test | train∩valid | test∩valid | Leak? |
|---|---|---|---|---|---|
| `01 Translate!` | 8075 / 1009 / 1010 | 4 | 8 | 2 | **YES** |
| `Data_Vanilla` | 7065 / 999 / 2030 | 3 | 12 | 0 | **YES** |

Exact-pair leakage is small, but it **understates the real problem**: with ~117 sentences per article and random row-level splitting, **effectively every article appears in all three splits.** The model trains on most of each article and is evaluated on held-out sentences from the _same_ article → inflated BLEU/chrF/COMET. The benchmark is not publishable until rebuilt.

## 3. Data quality

- `Data10KFinal`: 38 exact-duplicate rows (0.38%), 63 English-side duplicates → dedup.
- Mixed Arabic (`1.`) and Myanmar (`၁.`) numerals; stray internal whitespace; inconsistent step markers (`Part 1`, `Step 1`, `1`) → normalization (numeral rule shared with IFS-quantity).
- No `article_id` column anywhere → must be reconstructed.

## 4. Required rebuild (Phase 1 remaining work)

1. **Reconstruct `article_id`** for the 10,094 rows. Two options:
   - **(A, preferred)** rebuild canonical corpus from the per-article source files (`WikiHow-MTPE/*.xlsx`, `FORMAT/*.txt`) so `article_id = source file`; verify the union reproduces `Data10KFinal`.
   - **(B, fallback)** segment `Data10KFinal` at detected article-title/step-reset boundaries (heuristic; less reliable).
2. **Dedup** exact + near-duplicate rows.
3. **Normalize** (Unicode/NFC, numerals, whitespace, punctuation) via `myanmar-tools`.
4. **Article-disjoint split** — `GroupShuffleSplit` by `article_id`, **80/10/10, seed 42**, stratified by category where possible. Assert zero article overlap across splits.
5. **Re-run baselines** on the new splits (old `lab reports/` numbers are void).

## 5. Impact on the plan

- Phase 1 is ~60–70% done (data exists, is good, provenance known) but **not** done — the split rebuild is mandatory, not optional.
- Good news: the per-article source files exist, so option (A) is feasible and gives clean provenance + `article_id` for free.
