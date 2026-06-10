# Colab fine-tune runbook — NLLB-600M en→my (WikiHow-MY)

Goal of this session: produce the **fine-tuned** results row (chrF++ + COMET) and
beat the zero-shot baseline (**dev chrF++ > 36.01**).

## 0. What to upload to Google Drive

Upload the repo to `MyDrive/wikihow-my/` so the notebook's `PROJECT` path matches.
You need these (the English text in `data/processed/` is gitignored, so it is NOT on
GitHub — you must upload it from this local machine):

- `data/processed/train.jsonl` `dev.jsonl` `test.jsonl` ← the splits (required, gitignored)
- `src/` ← all training/inference/eval code
- `notebooks/colab_finetune_nllb.ipynb`

Fastest path: zip the three folders locally, upload the zip to Drive, unzip in a
Colab cell (`!unzip -q wikihow-my.zip -d /content/drive/MyDrive/wikihow-my`).

## 1. Runtime

Colab → Runtime → Change runtime type → **GPU**. A100 ideal; T4 works but:
if T4 OOMs during training, edit `src/train/config.yaml`:
`per_device_train_batch_size: 4` and `gradient_accumulation_steps: 16`.

## 2. Run the notebook cells in order

1. pip install (cell 1)
2. mount Drive + set `PROJECT` + `os.chdir` (cell 2) — edit `PROJECT` if your path differs
3. CUDA check (cell 3) — must print `CUDA: True`
4. **Fine-tune** (cell 5): `finetune_nllb.py` — early-stops on dev chrF, saves `/content/ckpt/best`
5. **Inference** (cell 7): writes zero-shot + fine-tuned test hyps
6. **Score + COMET + tables** (cell 9): now runs `--comet` (downloads `wmt22-comet-da`, ~2.3 GB, first call only)

## 3. Exit criteria (must all hold before declaring Phase 2 done)

- [ ] `experiments/results/main_results.json` has an `nllb_finetuned` block with a
      non-null `comet` value (and `comet_model: "Unbabel/wmt22-comet-da"`).
- [ ] Fine-tuned **dev** chrF++ > 36.01 (printed in the trainer's eval logs).
- [ ] Fine-tuned **test** chrF++ ≥ zero-shot test chrF++ (36.01). If it's _lower_,
      that's a red flag → check leakage / LR / epochs (plan §13), do not report it as-is.
- [ ] `paper/tables/main_results.tex` regenerated with the fine-tuned row.

## 4. Bring results back

Download from Drive (`MyDrive/wikihow-my/`) back to this machine, overwriting:

- `experiments/results/main_results.json`
- `experiments/results/nllb_finetuned_test_hyps.txt`
- `paper/tables/main_results.tex`

Optional: copy `/content/ckpt/best` to Drive if you want to keep the checkpoint
(not committed — it's large; rehydration release only ships data + code).

## Gotchas already handled

- COMET is **source-anchored** — the scorer reads the `en` field from the refs
  jsonl as the source. `--comet` is required; without it COMET is skipped (honestly
  noted, not silently null). Fixed 2026-06-07.
- All `src/` paths are portable (`__file__`-based), so `os.chdir(PROJECT)` is enough.
