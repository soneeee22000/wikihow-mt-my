# Project Spec — WikiHow en→my MT (two-track)

**Source of truth for phases + gates.** Detail lives in `execution-plan.md` (WMT track) and
`strategy-novelty-and-deployment.md` (Track-2 rationale + research findings). Log every run in
`experiments/LOG.md`. Update this file + project memory whenever a gate flips.

_Created 2026-06-08. Process model chosen by user: right-sized research spec (gate-by-gate; ECC
blueprint reserved for the Track-2 agent build)._

## Mission

Two-paper arc, not a single fine-tuning paper:

1. **WMT 2026 (floor, ~Aug):** the corpus + NLLB/Gemini/GTranslate benchmark + IFS-as-metric. Stakes
   the dataset + IFS claim publicly.
2. **ACL 2027 (flagship):** **IFS-as-a-control-signal** inside an agentic translate→reflect→repair
   loop for low-resource **procedural** text, + a narrow Burmese how-to deployment as real-world
   validation. The novel contribution.

## Success criteria (the gates that matter)

- **WMT:** leakage-free corpus + full benchmark table (chrF++/spBLEU/COMET, +MetricX) + IFS validated
  by human-followability correlation (or an honest mixed result) → submittable 6–10 pp.
- **Flagship:** IFS-guided agent beats fine-tuned NLLB and Gemini on **IFS and chrF/COMET** on a
  procedural test set, with human-followability validation + a live demo.

## Honesty guardrails (non-negotiable, every phase)

- Dataset = **MTPE / human post-edited**, never "gold-standard human translation."
- IFS claims bounded: step+action novel; entity+quantity **adopted** (cite M-ETA, WMT-CED, 2203.05227,
  2308.12674). "Agentic" alone is not the novelty — the structured procedural reward is.
- If fine-tuned **test** chrF++ < 36.01 (zero-shot) → **debug (leakage/LR/epochs), don't dress up.**
- Cite + cross-eval BURMESE-SAN; chrF++ primary (BLEU unreliable on unsegmented Burmese).

## Track 1 — WMT 2026 (floor)

| Phase | Feature                                               | Definition-of-Done gate                                                                 | Status                                                                                                                                                                                  |
| ----- | ----------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1    | Corpus + article-disjoint splits                      | 0 leakage, seeded, stats + datasheet                                                    | ✅ DONE (train 8302 / dev 908 / test 846)                                                                                                                                               |
| P2    | Benchmark: NLLB zero/FT, Gemini 2.5, Google Translate | all hyps scored chrF++/spBLEU/BLEU/COMET in `main_results.json`; FT dev chrF++ > 36.01  | ✅ DONE — 4 systems × 5 metrics. chrF++: GTrans 43.6 > Gemini-Flash 42.6 > FT 41.6 > ZS 36.0. COMET: **Gemini 0.904** > GTrans 0.894 > FT 0.886 > ZS 0.858 (metrics disagree on winner) |
| P3    | IFS metric + human validation                         | `ifs.py` ✅; IFS-vs-human-followability correlation table w/ significance vs chrF/COMET | 🟡 metric ✅; tooling ✅; **blind 4-system rating sheet built** (160 rows, `ratings_sheet.csv` + private `ratings_key.csv`, gitignored); ⬜ author + 2-3 rater ratings → `correlate.py` |
| P4    | Human eval + cultural analysis                        | ratings + Krippendorff α (or test-retest); error taxonomy                               | ⬜                                                                                                                                                                                      |
| P5    | Write-up + release                                    | 6–10 pp draft; rehydration release (CC BY-NC-SA); LaTeX tables auto-generated           | ⬜                                                                                                                                                                                      |

## Track 2 — ACL 2027 flagship (design in parallel; build after WMT submit)

| Phase | Feature                                                                          | Definition-of-Done gate                   | Status |
| ----- | -------------------------------------------------------------------------------- | ----------------------------------------- | ------ |
| F0    | Agent blueprint/spec                                                             | written spec (ECC `blueprint`/`prp-plan`) | ⬜     |
| F1    | IFS-as-reward agent loop (NLLB base + LLM reflect/repair, IFS gates refine+stop) | beats baselines on IFS + chrF on dev      | ⬜     |
| F2    | Narrow Burmese how-to deployment (1 category)                                    | live demo + real-user IFS ratings         | ⬜     |
| F3    | Human-followability validation + write-up                                        | correlation study + draft                 | ⬜     |

## Operating rules

- **One feature at a time, against its gate.** Don't jump ahead (we already broke this once: IFS built
  before P2 finished — acceptable but noted).
- **Log every run** in `experiments/LOG.md` (command, config, metric, sacreBLEU signature).
- **AskUserQuestion** for forks. **Commit only when asked.**

## Current focus

P2 baselines: (a) re-run Kaggle fine-tune (tokenizer-warning flood now fixed); (b) **Gemini 2.5 +
Google Translate baselines** — GPU-free, `src/infer/llm_baselines.py`, building now.
