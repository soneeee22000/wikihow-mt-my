# WikiHow-MY

**A human post-edited English→Myanmar instructional MT corpus, an NLLB-200 benchmark, and the Instruction Faithfulness Score (IFS).**

![license](https://img.shields.io/badge/code-Apache--2.0-blue)
![data license](https://img.shields.io/badge/data-CC%20BY--NC--SA%203.0-lightgrey)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![target](https://img.shields.io/badge/target-WMT%202026-success)

Burmese (Myanmar) is a low-resource language, and the procedural text people
actually need translated — _how-to_ instructions — is exactly where generic MT
breaks down: a dropped step, a swapped quantity, or a mistranslated entity can
make an instruction wrong rather than merely awkward. WikiHow-MY is a corpus,
benchmark, and metric built to study that problem head-on.

> **Honesty note.** This corpus is **human post-edited machine translation
> (MTPE)** — machine drafts that were edited and quality-checked by people
> (a paid professional translation service plus volunteer editors). It is
> _not_ presented as gold-standard from-scratch human translation, and the
> paper frames it as such.

---

## Three contributions

1. **A ~10K-pair en→my WikiHow instructional corpus** with **article-disjoint**
   train/dev/test splits — no how-to article spans two splits, so the benchmark
   measures generalization to _unseen instructions_, not memorization.
2. **An NLLB-200 benchmark** (zero-shot and fine-tuned) with chrF++ as the
   primary metric, alongside spBLEU and BLEU, plus LLM and Google Translate
   baselines.
3. **The Instruction Faithfulness Score (IFS)** — an evaluation metric centered
   on whether a translation preserves the **step + action** of an instruction
   (with entity and quantity preservation as adopted components), validated
   against human followability judgements.

## Pipeline

```mermaid
flowchart LR
    A["wikiHow how-to<br/>articles (EN)"] --> B["MT draft"]
    B --> C["Human post-edit + QC<br/>(MTPE: pro + volunteer)"]
    C --> D["Normalize + Zawgyi check<br/>+ exact-dedup"]
    D --> E["Article-disjoint splits<br/>(GroupShuffleSplit, seed 42)"]
    E --> F["Train / Dev / Test"]
    F --> G["NLLB-200 zero-shot<br/>+ fine-tune"]
    F --> H["Gemini / Google Translate<br/>baselines"]
    G --> I["chrF++ / spBLEU / BLEU<br/>/ COMET"]
    H --> I
    I --> J["IFS: step + action +<br/>entity + quantity"]
```

## Dataset at a glance

|                                   |                                          |
| --------------------------------- | ---------------------------------------- |
| Aligned pairs (after exact-dedup) | **10,056**                               |
| Distinct how-to articles          | **82**                                   |
| Train / Dev / Test (pairs)        | **8,302 / 908 / 846**                    |
| Train / Dev / Test (articles)     | **66 / 8 / 9**                           |
| Article overlap between splits    | **0** (asserted in code)                 |
| Median pairs per article          | 116                                      |
| Mean English length               | 14.3 whitespace tokens                   |
| Mean Myanmar length               | 93.0 characters (Myanmar is unsegmented) |
| Zawgyi-suspect rows               | **0** (clean Unicode)                    |

Myanmar has no orthographic word boundaries, so length is reported in
characters and metric choice favors segmentation-agnostic chrF++ over BLEU.

## Why article-disjoint splits matter

An earlier version of this project split the data at the **sentence level with
no fixed seed**, which leaked sentences from the same article across train and
test — inflating every score and making the benchmark meaningless. The current
splits group by `article_id` and assert zero article overlap. The full finding
is written up in [`docs/data-audit.md`](docs/data-audit.md); the fix lives in
[`src/data/build_splits.py`](src/data/build_splits.py) and reproduces
bit-for-bit with `seed=42`.

## Results (test set, n=846)

English→Myanmar. chrF++ is the primary metric.

| System                               |        chrF++ | spBLEU | BLEU | COMET |
| ------------------------------------ | ------------: | -----: | ---: | ----: |
| NLLB-200-distilled-600M (zero-shot)  |     **36.01** |  19.33 | 2.62 |     — |
| NLLB-200-distilled-600M (fine-tuned) | _in progress_ |        |      |       |
| Gemini 2.5 (few-shot)                |     _planned_ |        |      |       |
| Google Translate                     |     _planned_ |        |      |       |

The zero-shot number is on-anchor with published NLLB FLORES Burmese results,
which validates the harness. Tables are generated from
`experiments/results/main_results.json` by `src/eval/make_tables.py` — numbers
are never hand-typed into the paper.

## Project structure

```
wikihow-mt-my/
├── data/
│   ├── raw/         # source CSV + per-article text (gitignored — see Data & licensing)
│   ├── processed/   # train/dev/test .jsonl (gitignored); stats JSONs are tracked
│   └── sample/      # 20-row committed teaser
├── src/
│   ├── data/        # build_splits.py, normalize.py, stats.py
│   ├── infer/       # translate.py  (batched NLLB inference)
│   ├── train/       # finetune_nllb.py + config.yaml
│   └── eval/        # automatic.py (chrF++/spBLEU/BLEU/COMET), make_tables.py
├── notebooks/       # colab_finetune_nllb.ipynb  (GPU training)
├── experiments/     # LOG.md (append-only) + results/
├── paper/           # figures/ + tables/
└── docs/            # execution-plan.md, data-audit.md
```

## Getting started

```bash
# 1. install
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. rebuild the splits from source (needs data/raw/ — see Data & licensing)
python src/data/build_splits.py        # writes data/processed/*.jsonl, seed 42
python src/data/stats.py               # writes stats JSON + paper/figures/

# 3. zero-shot baseline (CPU works for a smoke test; GPU for the full run)
python src/infer/translate.py --split test --system nllb_zeroshot --limit 20
python src/eval/automatic.py \
    --hyps experiments/results/nllb_zeroshot_test_hyps.txt \
    --refs data/processed/test.jsonl --system nllb_zeroshot --limit 20

# 4. fine-tune on a GPU — run notebooks/colab_finetune_nllb.ipynb on Colab
```

## Data & licensing

The **code** is licensed under **Apache-2.0** (`LICENSE`). The **corpus** is
licensed under **CC BY-NC-SA 3.0** (`LICENSE-DATA`), inherited from wikiHow's
source license.

To respect that license, this repository does **not** redistribute the English
wikiHow text. The corpus is released by **rehydration** — the Myanmar
translations, the source article identifiers, and the build script are
published so the aligned pairs can be reconstructed locally; the raw and
processed parallel files are gitignored. A 20-row sample lives in
`data/sample/` for inspection. See `LICENSE-DATA` for full attribution.

## Status

Work in progress, targeting the **WMT 2026** research track. Done: corpus
cleaning, article-disjoint splits, dataset statistics, the evaluation harness,
and the NLLB zero-shot baseline. Next: NLLB fine-tuning + COMET, the
Gemini/Google Translate baselines, and the IFS metric with human validation.
Methodology and the running plan are in [`docs/execution-plan.md`](docs/execution-plan.md).

## License & author

- Code: Apache-2.0 — see [`LICENSE`](LICENSE)
- Data: CC BY-NC-SA 3.0 — see [`LICENSE-DATA`](LICENSE-DATA)
- English source content: © wikiHow (CC BY-NC-SA 3.0)

Author: **Pyae Sone (Seon)** — [@soneeee22000](https://github.com/soneeee22000)
