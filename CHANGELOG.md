# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this is a research artifact, so
"versions" track milestones rather than software releases.

## [Unreleased] — 2026-06-10

### Added

- **Full benchmark.** NLLB-200 fine-tuned, Gemini 2.5 Flash (5-shot), and Google
  Translate baselines on the WikiHow-MY test set, scored on chrF++/spBLEU/BLEU,
  COMET (`wmt22-comet-da`), and MetricX-24 (`hybrid-xl`). Fine-tuning improves every
  metric (+5.6 chrF++ in-domain).
- **FLORES+ out-of-domain evaluation** as a reference-bias control; the
  surface-vs-learned metric disagreement reproduces on independent references.
- **Instruction Faithfulness Score (IFS)** automatic components (`src/eval/ifs.py`)
  and the human-validation protocol + Label Studio config.
- **IFS+QE decoding-time reranking booster** (`src/rerank/`) with a beam/QE/IFS/
  fused/MBR/oracle ablation — reported as a negative result (the oracle beats Google
  Translate but no reference-free selector recovers the gain).
- **Published model:** [`PyaeSoneK/nllb-600m-wikihow-en-my`](https://huggingface.co/PyaeSoneK/nllb-600m-wikihow-en-my)
  on the Hugging Face Hub, with a metric-driven model card.
- **Compute orchestration** (`scripts/`): Kaggle GPU drivers for fine-tuning,
  N-best generation, and MetricX scoring; isolated-venv COMET/COMET-Kiwi scorers;
  `push_to_hub.py`.
- **Paper draft** (`paper/main.tex`): corpus, benchmark, FLORES+ bias control, IFS,
  reranking probe, with all tables generated from JSON by `make_tables.py`.

### Changed

- `src/eval/automatic.py`: real COMET behind `--comet` (was a no-op), source-anchored.
- Splits corrected to **article-disjoint** (seed 42); the earlier sentence-level
  split leaked articles across train/test (see `docs/data-audit.md`).

### Security / licensing

- English WikiHow source is never committed — corpus released by rehydration
  (Myanmar + URLs + build script); `.gitignore` blocks the parallel files, QE/MetricX
  inputs, and rating sheets that embed source text.

## [0.1.0] — 2026-06-06

### Added

- Initial WikiHow-MY corpus, article-disjoint splits, dataset statistics, the
  evaluation harness, and the NLLB-200 zero-shot baseline (chrF++ 36.01).
