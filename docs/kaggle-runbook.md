# Kaggle GPU runbook — NLLB-600M en→my fine-tune (standard workflow)

This is the **standard way to run training** for this project: headless on a free
Kaggle GPU (P100 / 2×T4, 30h/week), driven entirely from this machine via the
Kaggle API. No VS Code kernel bridge, no babysitting a browser tab.

## Why this and not Colab-in-VS-Code

The VS Code↔Colab kernel bridge is focus-dependent and drops every time you click
away — unusable for a multi-step training job. Kaggle's API lets us push a dataset +
a GPU kernel and poll for results, so the run is detached from the editor. See
`docs/colab-runbook.md` only as a manual fallback.

## One-time setup

1. Free Kaggle account → **Settings** → **API** → **Create New API Token**.
   This downloads `kaggle.json`.
2. Put it at `C:\Users\<you>\.kaggle\kaggle.json` (i.e. `~/.kaggle/kaggle.json`).
3. CLI is already installed (`pip install --user kaggle`).

## Smoke-first workflow (do this — don't blind-rerun the full job)

The expensive full run is ~1.5–2.5 h. **Never** debug it by re-running the full job and
discovering one bug per run. Validate the whole pipeline tiny+fast first:

```bash
# 0. (optional, free, ~5 min on CPU) local code check — needs ~8 GB RAM for NLLB-600M:
python src/train/finetune_nllb.py --smoke --out_dir /tmp/ckpt

# 1. Kaggle smoke (~10 min): runs the WHOLE remote pipeline tiny (64 train rows, 6 steps,
#    beam 1, 16 test sents) — catches env/GPU/dep bugs in one short run.
python scripts/kaggle_run.py --smoke

# 2. Only when the smoke is green, do the real run ONCE:
python scripts/kaggle_run.py
```

`--smoke` flips a constant in the pushed kernel and passes `--smoke`/`--limit` downstream;
fp16 auto-disables off-GPU. The smoke writes a (throwaway) `main_results.json` so a green
smoke is observable. For fast iterative debugging, a warm GPU box (RunPod/Lambda, ~$0.50/h,
install+download once, re-run in seconds) beats cold Kaggle kernels — use Kaggle for the
final free run.

## Run it (driven from here)

```bash
python scripts/kaggle_run.py
```

What it does:

1. **Dataset** — stages a _private_ dataset `<user>/wikihow-my-splits` containing
   the three `*.jsonl` splits (gitignored, never on GitHub) + `code.zip` (a snapshot
   of `src/`). Creates it, or pushes a new version on re-runs.
2. **Kernel** — pushes a GPU script-kernel `<user>/wikihow-my-finetune` (GPU +
   internet on) that runs `scripts/kaggle/run_kernel.py`: reconstructs the project,
   pip-installs deps, then fine-tune → inference (zero-shot + fine-tuned) → score
   (chrF++/spBLEU/BLEU/**COMET**) → LaTeX table.
3. **Poll** — prints status every 30 s until the kernel finishes.
4. **Pull** — downloads the artifacts into `experiments/results/` (JSON + hyps) and
   `paper/tables/main_results.tex`.

Other modes:

- `python scripts/kaggle_run.py --no-wait` — push and return immediately.
- `python scripts/kaggle_run.py --status` — re-poll the last kernel and pull when done.

## What comes back

- `experiments/results/main_results.json` — now with an `nllb_finetuned` block incl.
  non-null `comet` (`Unbabel/wmt22-comet-da`).
- `experiments/results/nllb_{zeroshot,finetuned}_test_hyps.txt`
- `paper/tables/main_results.tex`

The fine-tuned checkpoint stays on Kaggle (`/tmp` on the kernel) and is **not**
downloaded — it's ~2 GB and not needed for the paper (rehydration ships data + code).

## Exit criteria (Phase 2)

- [ ] `nllb_finetuned` row exists with non-null `comet`.
- [ ] Fine-tuned **dev** chrF++ > 36.01 (in the kernel log).
- [ ] Fine-tuned **test** chrF++ ≥ 36.01 (zero-shot). If lower → investigate
      (leakage / LR / epochs), do not report as-is.

## Iterating

Change data or hyperparameters locally, then re-run `python scripts/kaggle_run.py`.
It versions the dataset and re-runs the kernel — fully reproducible, same command
every time.

## Notes / gotchas

- Kaggle kernel limits: 9h/session, 12h max, 30h GPU/week — our run is ~1.5–2.5h.
- COMET downloads `wmt22-comet-da` (~2.3 GB) on the kernel; needs internet (enabled).
- If T4 OOMs, lower `per_device_train_batch_size` to 4 + `gradient_accumulation_steps`
  to 16 in `src/train/config.yaml`, then re-run.
