# IFS+QE Fused Reranking Booster (WMT-floor novelty)

Post-finetune, decoding-time booster that reranks the fine-tuned NLLB's N-best list by a
**fused utility = reference-free QE + structural IFS**, with the goal of matching/beating
Google Translate **in-domain** (WikiHow-MY test) without any extra training.

> **OUTCOME (2026-06-10): NEGATIVE RESULT — the "beat GT" thesis does not hold.** On the full
> test 16-best (n=846), the oracle (reference-chrF++ pick) reaches **46.27 chrF++, above Google
> Translate's 43.60** — a GT-beating candidate exists for ~93% of sources — but **no
> reference-free selector recovers it**: QE 40.90, IFS 40.88, fused (α=0.2) 40.92, MBR 41.35,
> vs top-beam 40.94. Root cause (diagnosed, on-thesis): COMET-Kiwi's per-source ranking
> correlates only **ρ=0.10** with reference chrF++ and agrees with the oracle pick 9.3% of the
> time (chance 6.2%); structural IFS is near-constant across the paraphrase beam. **Decision
> (user, architect): report as an honest negative finding that motivates the flagship**
> (structural reward as an in-loop signal, not a post-hoc reranker) — written into
> `paper/main.tex` §"Can Decoding-Time Reranking Close the Gap?" + `tables/rerank.tex`. **No
> parity-with-GT claim is made.** Numbers in `experiments/results/rerank_report.json`.
> The design below is retained as the record of what was built and tried.

## Why this is novel (white space)

Quality-aware decoding / N-best reranking by QE is established (Fernandes 2022; MBR, Freitag
2022; source-MBR, Lyu 2025). What is **not** in the literature is using a _structural
procedural-faithfulness_ signal (step / entity / quantity preservation — our IFS) as part of
the decoding-time utility. The booster is the first to fuse a learned QE estimator with a
structural instruction-faithfulness score for reranking, and it doubles as evidence that IFS
carries decoding-useful signal (a bridge to the Track-2 agentic work).

## Pipeline

```
fine-tuned NLLB ckpt ──beam(N)──► N-best candidates per source
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                    ▼                    ▼
              COMET-Kiwi QE         IFS (struct.)      (reference, oracle only)
              (ref-free, src+mt)    segment_ifs            chrF++ vs ref
                    │                    │
                    └────── per-source min-max normalize ──────┐
                                        │                       │
                          util = (1-α)·QE_norm + α·IFS_norm     │
                                        │                       │
                                     argmax ─────────► reranked hypothesis
```

1. **N-best generation** — `src/infer/translate_nbest.py`. Beam search with
   `num_beams=N`, `num_return_sequences=N` over the fine-tuned checkpoint. Runs locally
   (CPU, for smoke) or on Kaggle GPU (full). Writes
   `experiments/results/nbest/<split>_nbest.jsonl` = one row per source:
   `{"idx", "candidates": [N strings]}`.
2. **QE scoring** — `src/rerank/score_qe.py`. COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`,
   reference-free) over every (source, candidate) pair, in the isolated CPU comet venv
   (`C:\comet_venv`). Resumable. Writes `experiments/results/nbest/<split>_qe.jsonl`.
3. **Rerank + ablations** — `src/rerank/rerank.py`. Computes IFS per candidate (reuses
   `src/eval/ifs.py:segment_ifs`), normalizes QE+IFS per source, fuses, and emits one
   hypotheses file per variant so the existing `automatic.py` / COMET / MetricX harness
   scores them unchanged:
   - `rerank_beam1` — top beam (baseline; ≈ `nllb_finetuned`)
   - `rerank_qe` — argmax QE only (α=0)
   - `rerank_ifs` — argmax IFS only (α=1)
   - `rerank_fused` — argmax fused utility (α tuned on **dev**)
   - `rerank_oracle` — argmax by reference chrF++ (headroom upper bound; not a system)
4. **Eval** — chrF++ (primary) / spBLEU / BLEU via `automatic.py`; COMET via the
   flores-style CPU scorer; MetricX-24 via the Kaggle kernel. Compare every variant to
   `nllb_finetuned` and `gtranslate`.

## α tuning

α ∈ {0, 0.1, …, 1.0} grid, selected on the **dev** split by dev chrF++ (reference-based
tuning of a single hyperparameter on dev is standard and non-circular w.r.t. the test set).
Recorded in `experiments/results/rerank_report.json`.

## Circularity discipline (reviewer-facing)

The reranker uses COMET-Kiwi, which shares model lineage with our COMET eval. Therefore:

- **Primary claims** lead on **chrF++** (surface, fully independent of the reranker) and the
  **human followability** study.
- **MetricX-24** is an independent learned check (different family from COMET-Kiwi).
- **COMET** is reported but flagged as sharing lineage with the QE signal.
- `rerank_ifs` (argmax IFS) is **never** evaluated by IFS (fully circular) — only by
  chrF++/COMET/MetricX/human.
- `rerank_oracle` quantifies how much of the gap to GT is _reachable_ from the N-best list at
  all; if oracle ≤ GT, no reranker can close it and we say so.

## Scope of the claim

In-domain (WikiHow-MY) only. The FLORES+ gap to GT is large (+8.8 chrF++), so we make **no**
general-parity claim — the booster is a domain-specific enhancement, consistent with the
domain-generalization finding.

## Smoke-first

`--smoke` / `--limit 8` runs the entire chain locally on CPU with the real local checkpoint
(`checkpoints/nllb_finetuned_wikihow/`) in minutes, validating generation → QE → IFS →
fuse → eval before any full or Kaggle run.

## Compute

- N-best generation: Kaggle GPU for the full test+dev run (chosen); local CPU is also viable
  since the checkpoint is local and QE is CPU-bound regardless.
- QE: local CPU comet venv (COMET-Kiwi); does not install cleanly on Kaggle's Python 3.12
  (same pytorch-lightning pin issue as COMET), so it stays local.
- MetricX of the winning variant(s): existing Kaggle MetricX kernel.
