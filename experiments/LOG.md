# Experiment Log (append-only)

## 2026-06-06 — Phase 1 data audit (read-only)

- Ran `_phase1_audit.py` over all candidate CSVs + split notebooks.
- **Canonical = Data10KFinal.csv (10,094 rows, ≥86 articles).**
- Provenance: ~5K professional (BharSarPyan) + ~6K volunteer MTPE → merged/deduped. Fully MTPE pipeline.
- **Split FAIL:** existing train/test/valid are sentence-level random, unseeded → article leakage + 14–15 exact-pair leaks. Existing lab-reports baselines VOID.
- Quality: 38 exact-dup rows, mixed numerals, stray whitespace; no article_id column.
- Decision: rebuild article-disjoint splits (GroupShuffleSplit by article_id, 80/10/10, seed 42) before any training.
- Artifacts: `Phase1_Data_Audit_2026-06-06.md`, `_phase1_audit_report.json`.

## 2026-06-06 — Phase 1 rebuild: article-disjoint splits

- `src/data/build_splits.py`: reconstructed article_id via title-boundary detection (each t*.txt/FORMAT/*.txt = 1 article; Data10KFinal is in article order).
- 10,094 → 10,056 after dropping 38 exact-dup pairs.
- 82 articles detected (median 116 sentences/article, min 31, max 522).
- Splits (seed 42, GroupShuffleSplit by article_id): train 7827 / dev 908 / test 1321; articles 65/8/9; **article overlap = 0 (asserted).**
- Output: data/processed/{train,dev,test}.jsonl + split_report.json.
- OPEN: 475 leading rows (~4 articles) bucketed as article 0 (titles unmatched to a source file) -> all forced to train. Tighten by recovering those boundaries.
- NEXT: Myanmar normalization (numerals/whitespace/Zawgyi check), category labels + stats figure, then re-run NLLB baselines on clean splits (old lab-reports numbers void).

### refinement (same day)

- Forced the 475-row unrecoverable-boundary residual into TRAIN; dev/test now contain ONLY cleanly-bounded articles.
- Final splits (seed 42): train 8302 / dev 908 / test 846; articles 66/8/9; overlap 0.

## 2026-06-06 — Phase 1 normalization + stats (COMPLETE)

- normalize.py: NFC + whitespace + Myanmar-punct spacing, applied BEFORE dedup/split. Numerals untouched (eval-time concern).
- Zawgyi check (corrected heuristic: U+1031 not preceded by consonant/medial): **0 suspect rows — corpus is clean Unicode.** (Confirm via myanmar-tools in Colab.)
- Numeral profile (MY side): none 8625 / myanmar 998 / arabic 458 / mixed 13 -> motivates IFS cross-script numeral normalization.
- stats.py -> data/processed/dataset_stats.json + paper/figures/dataset_stats.png.
  - 10,056 pairs; 82 clean articles; 116.8 sentences/article (median 116, max 274).
  - EN 14.3 tok mean (median 13, p95 28); MY 93 chars mean (median 85, p95 185).
- Phase 1 deliverables DONE: article-disjoint seeded splits + stats + figure.
- OPEN polish (optional): drop len==1 step-number-only rows; recover 4 residual articles (82->86); category labels need source URLs.
- NEXT: Phase 2 — Colab NLLB fine-tune + eval harness (chrF++/spBLEU/COMET) on clean splits.

## 2026-06-06 — Phase 2 scaffold + zero-shot smoke test

- Eval harness src/eval/automatic.py: chrF++ (primary), spBLEU (flores200 tok), BLEU, COMET hook (lazy; Colab/GPU). sacreBLEU signatures reported.
- src/infer/translate.py: batched NLLB en->my inference (GPU or CPU), forced_bos mya_Mymr.
- src/train/{config.yaml,finetune_nllb.py}: HF Seq2SeqTrainer, LR 3e-5, early-stop on dev chrF, seed 42.
- notebooks/colab_finetune_nllb.ipynb: end-to-end Colab flow (train -> infer -> score -> table).
- src/eval/make_tables.py: auto-generates paper/tables/main_results.tex (no hand-typed numbers).
- requirements.txt added.
- SMOKE TEST (20 test sents, CPU, zero-shot): chrF++ 36.55 / spBLEU 12.61 / BLEU 5.36 -- on-anchor with NLLB FLORES (~34 chrF++). Pipeline validated end-to-end.
- All scripts py_compile clean; config + notebook JSON valid.
- IN PROGRESS: full 846-sent zero-shot test inference (CPU, background) -> will score for real baseline row.
- NEXT: score full zero-shot; fine-tune on Colab; then Gemini + Google Translate baselines; then Phase 3 IFS.

## 2026-06-06 — REAL zero-shot baseline (full test, article-disjoint)

- nllb_zeroshot on full test (n=846): **chrF++ 36.01 / spBLEU 19.33 / BLEU 2.62** (BLEU low = expected on unsegmented Myanmar; chrF++/spBLEU are the meaningful metrics).
- On-anchor with NLLB FLORES (~34 chrF++). First valid, leakage-free baseline row -> main_results.json.
- Smoke-test entry (n20) removed from results.
- NEXT: Colab fine-tune (must beat 36.01 chrF++ on dev) + COMET; then Gemini 2.5 + Google Translate.

## 2026-06-07/08 — Kaggle GPU workflow, real COMET, IFS metric, strategy pivot

- Training moved to **Kaggle, API-driven** (Colab/VS-Code bridge abandoned): `scripts/kaggle_run.py` (orchestrator), `scripts/kaggle/run_kernel.py` (remote pipeline), `docs/kaggle-runbook.md`. Code+splits ride in ONE private Kaggle dataset (no git push; English stays off GitHub).
- Kaggle env fixes (each peeled one failure layer): legacy `kaggle.json` auth (new KGAT token unsupported by installed SDK); SDK single-ref signatures (`kernels_status(ref)` etc.); UTF-8 `open` monkeypatch (Burmese logs crashed cp1252); **log/output-based completion** (Kaggle `status` endpoint 500s); recursive dataset staging (mount is `/kaggle/input/datasets/<user>/<slug>/`, racy → retry+settle); **fresh kernel slug** (`wikihow-my-ft`) after a UI edit wiped `dataset_sources=['']`; **P100 sm_60** unsupported by Kaggle torch → install `torch 2.4.1 + torchvision 0.19.1 + torchaudio 2.4.1` (cu121); **transformers pinned 4.46.3** (>=4.50 blocks `torch.load` on torch<2.6, CVE-2025-32434; NLLB has no safetensors); **COMET decoupled** (`unbabel-comet` pins ancient pytorch-lightning, breaks pip) → compute COMET locally from pulled hyps; **OOM** at batch 8 → batch 4 / eval-batch 2 + `expandable_segments`; tokenizer-deprecation warning flood suppressed in `finetune_nllb.py`.
- `src/eval/automatic.py`: real COMET behind `--comet` (was a silent no-op).
- **`src/eval/ifs.py` (NEW):** Instruction Faithfulness Score, source-anchored. Automatic components quantity (Arabic+Burmese digits), entity (brands/acronyms/URLs/proper nouns), step (clause counts); action reserved for the human protocol. Unit-tested; **human-reference ceiling = IFS 93.86** (quantity 97.46 / entity 93.06 / step 88.27) over the 846-seg test set.
- Training reached **real training** (186 steps before OOM, since fixed). Latest run's committed output is empty (possibly stalled at first eval / interrupted) — **no fine-tuned result yet**; clean re-run pending.
- **STRATEGY PIVOT** (`docs/strategy-novelty-and-deployment.md`, `docs/spec.md`): two-paper arc — WMT 2026 floor + **ACL 2027 flagship = IFS-as-control-signal agentic translation**. Narrow Burmese how-to deployment. Corpus is pre-24-Mar-2025 → CC BY-NC-SA clear. wikiHow has no Burmese edition.

## 2026-06-08 — P2 feature: Gemini + Google Translate baselines (GPU-free)

- `src/infer/llm_baselines.py`: REST-based (no SDK version risk), `--system gemini|gtranslate`, Gemini 2.5 5-shot (seed-42 demos) / Cloud Translation v2. Writes `experiments/results/<system>_test_hyps.txt` in the harness format; resume-safe; newline-sanitised for alignment. Keys via `GEMINI_API_KEY` / `GOOGLE_TRANSLATE_API_KEY`.
- BLOCKER: API keys (user). Then score with `automatic.py` + `ifs.py` for the benchmark rows.

## 2026-06-08 — P3 feature: IFS human-validation backbone (GPU-free)

- `docs/ifs-human-validation-protocol.md`: rubric (Adequacy + Instruction-Followability 1-5 + binary component checks incl. the human-only **action**), sampling (K source sents x systems, blind+shuffled), reliability (author full set + 2-3 light raters on ~50-row overlap -> Krippendorff α + author test-retest), analysis plan, P3 gate.
- `src/eval/correlate.py`: `--make-sheet` (blind shuffled ratings sheet across systems) + analysis (per-segment IFS / sentence chrF++ / sentence BLEU / optional COMET column vs human followability; Pearson + Spearman; **Williams test** comparing IFS against each metric on the shared human var). Uses scipy 1.15.
- Dry-tested end-to-end on synthetic ratings: correlations + Williams compute correctly (IFS-vs-BLEU p=0.042 IFS-higher on synth data); test artifacts removed.
- DOWNSTREAM: needs all-systems hyps (P2: FT + Gemini + GTranslate) before building the real sheet + collecting ratings.

## 2026-06-08 — WORKFLOW FIX: smoke-first (stop blind full re-runs)

- Root-caused the 2-day thrash: we debugged a 2-3h cold batch job by re-running it and finding ONE bug per run (tokenizer-kwarg -> deps -> P100 -> OOM -> torch.load). Anti-pattern. Fix = validate the whole pipeline tiny+fast before spending GPU hours.
- Added `--smoke` end-to-end: `finetune_nllb.py --smoke` (64 train / 16 dev rows, max_steps 6, batch 2, beam 1, gen 32, fp16 auto-off on CPU); `run_kernel.py` SMOKE constant (limits infer + scores on 16); `kaggle_run.py --smoke` flips the kernel constant before push. Compiles + arg/flip verified.
- New workflow (docs/kaggle-runbook.md): (0) optional local CPU smoke for code bugs; (1) `kaggle_run.py --smoke` ~10 min to catch env/GPU/dep bugs in ONE run; (2) full run ONCE when green. Warm GPU box noted as the fast-iteration upgrade.
- NEXT: run the Kaggle smoke once -> green -> one full fine-tune run for the real headline.

## 2026-06-08 — P2 HEADLINE: fine-tuned NLLB result landed (the stuck run actually finished)

- The "stuck" full run (b1dstbsw2) wasn't stuck — it was just slow (eval flood); the local poller stayed alive and pulled. Committed Kaggle output is only visible at completion, which is why API checks showed empty mid-run. (Smoke-first still right going forward.)
- **nllb_finetuned (full test, n=846, article-disjoint): chrF++ 41.64 / spBLEU 23.18 / BLEU 3.74** vs zero-shot 36.01 / 19.33 / 2.62 -> **+5.63 chrF++ / +3.85 spBLEU. Honesty gate cleared** (well above 36.01; not dressed up). Pulled: main_results.json, main_results.tex, both \*\_test_hyps.txt.
- IFS scored (ifs_results.json): zero-shot 94.11 (q96.99/e94.32/s87.93); finetuned 95.79 (q99.59/e94.12/s91.52); human-ref ceiling 93.86. **CAVEAT (honest):** both MT systems are AT/ABOVE the human-ref ceiling on automatic IFS -> not "MT more faithful than humans" but the metric rewarding literal surface preservation (numerals/clause counts copied mechanically) that a fluent human post-edit localizes away. High auto-IFS != followability. Range is also compressed (94->96). => human-correlation study (P3) is mandatory before any IFS quality claim; do NOT report IFS as a quality win yet.
- OPEN P2: COMET (run locally on pulled hyps; unbabel-comet install may need care), Gemini + GTranslate baselines (keys).

## 2026-06-08 — P2: COMET scored (isolated venv, smoke-first)

- `scripts/run_comet.py` (standalone, merges into main_results.json without touching other metrics). Ran in an ISOLATED venv (`C:\comet_venv`, torch 2.12.0+cpu) so unbabel-comet never touches the main env — clean venv resolved with NO dependency hell (vindicates the Kaggle COMET-decoupling decision). Smoke-first: import check -> 4-seg smoke (model ~2.3GB cached) -> full.
- **COMET (wmt22-comet-da, n=846): zero-shot 0.8579 -> fine-tuned 0.8863 (+0.028).** Fine-tune wins on ALL metrics (chrF++/spBLEU/BLEU/COMET) — consistent, robust improvement.
- `make_tables.py` extended (IFS column + ifs_components + ifs_correlation tables) and naming fixed (gemini/gtranslate). main_results.tex + ifs_components.tex regenerated from JSON (no hand-typed numbers).
- P2 NLLB systems COMPLETE. Remaining P2: Gemini 2.5 + Google Translate rows (need API keys). Then P3 human study.

## 2026-06-08 — Colab notebook made self-contained + publishable (COMET inline + IFS)

- `notebooks/colab_finetune_nllb.ipynb` reworked so COMET is no longer null: train deps at top (NO unbabel-comet); train -> infer -> score chrF++/spBLEU/BLEU + **IFS** (added) -> new **§3b** installs unbabel-comet **AFTER training** (so its pins can't break training; Colab T4 sm_75 has no P100/torch-downgrade issue) -> `scripts/run_comet.py` merges COMET -> make_tables -> final results-table cell. Removed broken `myanmar-tools` install + stray CPU diagnostic cell. Valid JSON, 14 cells, install-after-train ordering asserted.
- `src/infer/llm_baselines.py`: added `.env` loader (gitignored) + `.env.example` so Gemini/GTranslate keys never get pasted into chat.
- NOTE: notebook is structurally correct for Colab but not executed here (user accepted that with the self-contained choice).

## 2026-06-08 — P2 COMPLETE: Gemini + Google Translate baselines (GCP set up via gcloud)

- User had deleted GCP projects (cost-averse) but the **billing account survived**. Set up both baselines entirely from the gcloud CLI using the existing open billing account `0171A7-...`: created project `wikihow-mt-baseline-pyae`, linked billing, enabled Cloud Translation + Generative Language APIs, minted two restricted API keys (translate-only, gemini-only) → written to `.env` (gitignored, never printed).
- **Google Translate**: official Cloud Translation v2, free tier (500k chars/mo) covers our ~68k → **$0**. **Gemini**: gemini-2.5-flash, paid tier ~**$0.50** (user's old key was an invalid `AQ.A...` OAuth-ish string, rejected as both key and Bearer; minted a proper `AIza...` key instead).
- **FINAL 4-system benchmark (test n=846, article-disjoint):**
  - chrF++ / spBLEU / BLEU / COMET / IFS
  - nllb_zeroshot 36.01 / 19.33 / 2.62 / 0.8579 / 94.11
  - nllb_finetuned 41.64 / 23.18 / 3.74 / 0.8863 / 95.79
  - gemini-2.5-flash 42.59 / 24.77 / 3.47 / **0.9037** / 94.33
  - gtranslate **43.60 / 27.65 / 5.07** / 0.8942 / 95.27
- **KEY FINDING (honest, paper-worthy): metrics disagree on the winner.** Surface metrics (chrF++/spBLEU/BLEU) → Google Translate best; semantic COMET → Gemini Flash best (0.9037). Consistent with BURMESE-SAN (Gemini leads on learned metric MetricX). Google Translate beats fine-tuned NLLB on all surface metrics. Gemini = **Flash** (cost), weaker than Pro — label precisely in paper.
- P2 DONE. Tables regenerated (main_results.tex, ifs_components.tex). NEXT: P3 human study (build 4-system rating sheet → recruit raters → correlate.py). Optional: FLORES+ cross-eval + MetricX (BURMESE-SAN defense); a Gemini-2.5-Pro row if budget allows.

## 2026-06-08 — P3 human-rating set up as a web tool (Label Studio on Railway)

- Decision: CSV rating → too much friction; collect via a link instead. Chose **Label Studio** (off-the-shelf, citeable, blinding/multi-rater/agreement/export) over a custom app (speed/rigor on deadline). Host = **Railway** (~$5/mo; persistent Postgres so ratings can't be lost mid-eval; free tiers rejected for fragile persistence).
- Built **truly-blind** sheet: `correlate.py --make-sheet` now writes `ratings_sheet.csv` (opaque `seg-NNNN` ids, ONLY src_en+hyp_my) + private `ratings_key.csv` (id→system,ref_my). Fixes earlier non-blind leak (segment_id/system/ref were visible). `analyze()` rewritten to join the key by id. Round-trip tested.
- LS artifacts: `experiments/results/ratings_ls_import.csv` (160 blind tasks), `ratings_ls_config.xml` (followability+adequacy 1-5 UI), `scripts/ls_export_to_csv.py` (LS JSON export → ratings_filled.csv, keeps annotator for Krippendorff α). Runbook: `docs/labelstudio-railway.md`.
- Data protection: `.gitignore` now ignores `experiments/results/ratings*.csv` (English source stays local); key file NOT uploaded to Label Studio (keeps study blind); keep to the 160-item subset.
- HANDOFF (user): deploy LS on Railway per runbook → import csv + paste config → recruit Burmese raters (SpeakProof/FB/Telegram), first ~50 tasks to 2-3 raters for α → export → `ls_export_to_csv.py` → `correlate.py`. That correlation = the paper's IFS validation.

## 2026-06-08 — Research-framing pass: 5-area literature review (the scholarly spine)

- User flagged the real gap: over-indexed on engineering (Kaggle/GCP/Label Studio plumbing), under-invested in scholarship (no systematic lit review, no RQs, Related Work). Correct. Ran 5 parallel cited-research agents → synthesized into **`docs/literature-review.md`** (Related Work by area + novelty verdict + RQs/hypotheses + corrected contributions + eval matrix + verification gaps).
- **Novelty corrections (honesty-critical):** corpus novelty HOLDS (WikiLingua has no Burmese; ALT is news). IFS entity+quantity NOT novel (cite WMT-CED NUM/NAM, M-ETA, ACES). **RecipeGen (2506.06733) is the closest pre-empt** (action/step/ingredient/quantity metrics for recipe _generation_) → step+action = "first in MT, to our knowledge" + credit it. Cleanest novelty = IFS validated vs human **followability** in low-resource Burmese (COMET miscalibrated: Falcão/AfriCOMET/SSA-COMET). Construct = task-based MT eval × skopos; disambiguate from IFEval/XIFBench. Flagship in-loop-reward white-space CONFIRMED (differentiate from TEaR/LLMRefine/Ng/MT-R1/TAT-R1/DelTA).
- **Stats plan locked:** segment-level Spearman(primary)+Pearson + 1000× bootstrap CIs + Williams test; ordinal Krippendorff α on ≥50 overlap + test-retest + ~20% QC; no system-level (Mathur 2020).
- **NEW action items:** add FLORES+ cross-eval + MetricX-24; `correlate.py` add bootstrap CIs + Spearman-primary; update protocol doc + seed QC items in rating sheet. Verification gaps: OPUS/HF search for stray wikihow-my; pull NLLB FLORES en→my numbers.

## 2026-06-08 — Paper drafting started (LaTeX in-repo, P5)

- Chose to draft in LaTeX in-repo over external tools (only ACL-valid format; full context here). `paper/main.tex` (portable article class — compiles now; swap to official ACL template for submission) + `paper/references.bib` (~30 entries seeded from the lit review; VERIFY vs ACL Anthology before camera-ready).
- Drafted the stable sections with real numbers traceable to JSON via `\input{tables/*}`: Abstract, Intro (with followability-vs-IFEval scope clarification), Related Work (all 5 areas), Corpus (§3, exact stats + MTPE + rehydration/license), Benchmark (§4, the metrics-disagree finding), IFS definition (§5, honest novelty framing + RecipeGen/WMT-CED/M-ETA credit), Human Validation method (§6, stats plan), Limitations, Conclusion.
- **TODO placeholders** (await data): IFS-vs-human correlation results (§6), Analysis (§6 cultural/error taxonomy), final Abstract/Intro/Conclusion numbers; FLORES+/MetricX rows.
- Builds clean: `cd paper && pdflatex main && bibtex main && pdflatex x2` → **main.pdf, 6 pages, 0 undefined citations, 0 bib warnings**. LaTeX build artifacts gitignored.

## 2026-06-08 — Overnight FLORES+ re-train launched (--no-wait); morning handoff
- Smoke (`kaggle_run.py --smoke`) went GREEN: train -> WikiHow infer/score -> FLORES+ infer/score (bundled flores.jsonl) -> PIPELINE DONE. Caught + fixed the FLORES gating bug (both openlanguagedata/flores_plus and facebook/flores are HF-gated) by bundling FLORES-200 devtest (1012 pairs) into the private dataset (data/processed/flores.jsonl, gitignored; added to SPLITS).
- SMOKE CLOBBER BUG found + fixed: smoke's n=16 main_results.json overwrote the real 4-system file. Rebuilt real benchmark from intact hyps (zero 36.01/.858, FT 41.64/.886, gemini 42.59/.904, gtrans 43.60/.894, all n=846). Orchestrator fix: smoke no longer distributes; real-run pull now MERGES main_results.json (preserves gemini/gtranslate, adds flores rows) instead of overwriting.
- run_kernel.py now PERSISTS the checkpoint to /kaggle/working/ckpt_best (full mode) — fixes the lost-checkpoint sloppiness.
- LAUNCHED full overnight run via `python scripts/kaggle_run.py --no-wait` (kernel pyaesonekyaw/wikihow-my-ft, server-side, ~2-3h). It re-trains -> saves checkpoint -> infers+scores WikiHow test + FLORES+ devtest (zero/FT).
- **MORNING TASK:** `python scripts/kaggle_run.py --pull` (merges nllb + flores rows into main_results.json, preserving gemini/gtranslate) -> `python src/eval/make_tables.py` -> verify checkpoint persisted in kernel output (kernels_output file list, ckpt_best/) -> report FLORES+ domain-gap numbers. NOTE: a --pull will also download the ~2GB checkpoint to temp (kernels_output is all-or-nothing) — acceptable once, or verify-only via file-list. STILL PENDING: MetricX (local venv, transformers 5.x too new -> pin 4.x); P3 human study (Label Studio on Railway).

## 2026-06-08 (morning) — Overnight FLORES+ run LANDED ✓
- Kernel pyaesonekyaw/wikihow-my-ft full run COMPLETED (log: "PIPELINE DONE", "saved checkpoint -> ckpt_best/"). `kernels_status` still 500s; `kernels_list_files` returns stub ~880B sizes (unreliable) — verified via real `kernels_output` download (2.5GB total).
- **Checkpoint PERSISTED** (rule #6 satisfied): model.safetensors 2.46GB + tokenizer/configs -> moved to `checkpoints/nllb_finetuned_wikihow/` (gitignored). Re-train no longer required for future inference.
- WikiHow test reproduced EXACTLY (deterministic): nllb_zeroshot 36.01 / nllb_finetuned 41.64 chrF++.
- **FLORES+ devtest cross-eval (out-of-domain, n=1012):** nllb_zeroshot_flores **29.17** chrF++ (spBLEU 14.79) / nllb_finetuned_flores **33.50** chrF++ (spBLEU 17.78).
- **Finding:** WikiHow fine-tune helps BOTH domains — in-domain +5.63, out-of-domain FLORES+ **+4.33** chrF++. No catastrophic forgetting; the procedural fine-tune generalizes to general-domain text. NLLB also scores higher in-domain (WikiHow 36.01) than on general FLORES (29.17).
- Merged add-only into main_results.json (6 systems; COMET on the 4 in-domain rows preserved). Tables regenerated. FLORES COMET computing via isolated venv (scripts/run_comet_flores.py).
- MORNING REMAINING: confirm FLORES COMET merged -> regen tables -> add FLORES+ rows to paper §4. Then MetricX (transformers pin) + P3 human study (Label Studio/Railway).

## 2026-06-08 — MetricX-24 scoring (BURMESE-SAN comparability) — in progress
- Decision: MetricX-24 **hybrid-XL** (reference-based), the variant size balancing strength vs Kaggle feasibility (user-chosen). Error score [0,25], LOWER better. All sizes use the `google/mt5-xl` tokenizer.
- New code: `src/eval/build_metricx_inputs.py` (builds per-system source/hypothesis/reference jsonls -> `experiments/results/metricx_inputs/`, **gitignored — contains English source**), `scripts/kaggle/run_metricx.py` (GPU kernel: pip transformers==4.46.3 + sentencepiece/accelerate, git-clone google-research/metricx, run `metricx24.predict` per system via PYTHONPATH, mean prediction -> metricx_results.json), `scripts/kaggle_metricx.py` (orchestrator, reuses kaggle_run helpers; merges `metricx24` into main_results.json without clobbering).
- Smoke-first: smoke uses the small `large` model + 8 rows + 2 systems to validate the path cheaply; full uses XL on all 6 systems.
- Bugs caught by smoke before the real run: (1) dataset `dir_mode="skip"` dropped the `metricx_inputs/` subfolder -> stage jsonls at dataset ROOT. (2) **409 Conflict**: kernel and dataset can't share a slug (shared user URL namespace) -> KERNEL_SLUG=`wikihow-metricx-run` vs DATASET_SLUG=`wikihow-my-metricx`.
- Paper: make_tables.py now emits a `MetricX$\downarrow$` column in the main table and the FLORES domain table (swapped spBLEU out of the FLORES table for width). Renders `--` until scores land.

## 2026-06-08 — MetricX-24 COMPLETE ✓ (smoke-validated, full XL run merged)
- Smoke (large model, 8 rows, 2 systems) went GREEN after 3 fixes; confirmed transformers==4.46.3 runs MT5ForRegression (no need for repo's old 4.30.2 pin). 4th distinct bug caught by smoke: transformers.Trainer auto-inits wandb on Kaggle -> set WANDB_DISABLED/WANDB_MODE=disabled in the predict env.
- **Full MetricX-24 hybrid-XL (reference-based) scores merged into main_results.json (lower=better):** nllb_zeroshot 4.09 / nllb_finetuned 3.23 / **gemini 2.39** / gtranslate 2.71; FLORES+ nllb_zeroshot 5.64 / nllb_finetuned 4.80.
- **KEY (sharpens the paper):** BOTH learned metrics now rank Gemini #1 (COMET 0.904, MetricX 2.39) while ALL surface metrics rank Google Translate #1 (chrF++ 43.60). Two independent learned metrics agreeing => the split is genuinely surface-vs-semantic, not a COMET artifact; matches BURMESE-SAN (Gemini tops MetricX). Fine-tuning improves MetricX in both domains (4.09->3.23 in-domain, 5.64->4.80 FLORES+).
- Paper §4 updated: MetricX column in main + FLORES tables (make_tables.py), Findings + domain-generalization prose rewritten around the two-learned-metrics-agree result, metric def + \citep{metricx2024} added. **paper compiles clean: 7 pages, 0 undefined citations, PDF built (MiKTeX pdflatex+bibtex).**
- Gitignored: experiments/results/metricx_inputs/ (English source) + experiments/results/*.log (Kaggle logs may echo source).
- REMAINING: P3 human-followability study (Label Studio/Railway, user action). MetricX + FLORES+ benchmark fully done.

## 2026-06-08 — Reference-bias investigation (search-first + 3 agents)
- TRIGGER: hypothesis that GT "wins" surface metrics because the MTPE references were post-edited from Google Translate output.
- **Empirical signature test (scripts/_bias_signature.py, n=846): REFUTED the strong form.** Exact-match-vs-ref rates: nllb_ft 0.47%(4) > gemini 0.24%(2) = gtranslate 0.24%(2) > nllb_zs 0%. GT is NOT the leader; a GT-base would exact-match dozens-hundreds. Ref resembles GT output no more than gemini/nllb do (gemini is closer to GT output than the ref is). GT's ~1-2 chrF++ lead = genuine quality, not GT-reference contamination. Caveat: only rules out GT-as-verbatim-base; heavy editing or a DIFFERENT base MT undetectable here; suggest NFC re-check (low priority — means show no anomaly).
- **Provenance archaeology: base MT system is UNRECORDED anywhere** (archive + repo). 6K volunteer = "post-editing of MT" (base unnamed); 5K BharSarPyan = professional but post-edited (file misleadingly named human_written_data_5K.csv). No from-scratch human subset exists; entire corpus is MTPE. -> disclose base-MT-unknown in Limitations.
- **Literature (citeable):** post-editese = exacerbated translationese (Toral 2019 MT Summit); Freitag et al. 2020 EMNLP "references are not innocent" (paraphrased refs; all metrics incl. COMET affected); Zhang&Toral 2019 WMT (translationese inflates scores, shifts rankings, worse for low-resource); Toral et al. 2018 (human-parity collapses on original-source). Mitigations: reference-free QE (COMET-Kiwi, MetricX-QE), paraphrased/multi refs, FLORES as independent control.
- **Post-finetune "beat-GT" techniques (citeable):** quality-aware decoding/N-best rerank (Fernandes 2022 NAACL), MBR w/ neural metric (Freitag 2022 TACL; Eikema&Aziz 2020), source-based MBR (Lyu 2025 ACL), TEaR self-refine (Chen 2025 NAACL-F), GRRM reward (2026), instruction-tuning, LLM->NMT distillation. **WHITE SPACE: no one uses a structural instruction-faithfulness signal (IFS) as the decoding/rerank utility** -> unifies WMT floor booster + ACL flagship. Risk: IFS rewards literalness -> use FUSED QE+IFS utility, evaluate on non-reranked metrics + human (avoid circularity).
- **DECISIVE NEXT EXPERIMENT (proposed): run GT + Gemini on FLORES+ (independent refs) -> compare GT-vs-fineNLLB gap on FLORES+ vs WikiHow. Settles quality-vs-bias. ~$0.50.**

## 2026-06-08 (resume after laptop close)
- Laptop close killed the CPU COMET process mid-run (49/64); COMET has no mid-run checkpoint so gtranslate_flores.comet stayed null.
- Re-ran: GT FLORES+ COMET = **0.8952** (full 64/64). Bug: empty gemini_flores_hyps.txt (0 B, created by the 429-failed Gemini run) passed the os.path.exists guard -> COMET tried to score an empty list and crashed BEFORE json.dump, so the computed 0.8952 wasn't persisted. Fixed run_comet_flores.py guard to also skip 0-byte files; merged 0.8952 directly (valid full-batch score, no recompute).
- Tables regenerated; paper recompiles clean (7 pages, 0 undefined citations).
- Still pending: GT FLORES+ MetricX (Kaggle GPU); entire Gemini FLORES+ row blocked on Gemini spend cap (429) -> user must raise cap at ai.studio/spend.

## 2026-06-09 (FLORES+ bias-control table COMPLETE)
- Gemini spend cap raised -> generated Gemini FLORES+ hyps (1012, gemini-2.5-flash). chrF++ 39.49 / spBLEU 24.62 / BLEU 3.56 / COMET 0.8998.
- GT FLORES+ COMET 0.8952 (recovered + saved).
- MetricX bug: kernel had a HARDCODED full-mode system list (6 old systems) -> crashed on pruned nllb_zeroshot.jsonl. Fixed run_metricx.py to discover systems from staged dataset files. Re-ran: GT_flores MetricX 3.3128, Gemini_flores MetricX 3.0787.
- Full FLORES+ table (n=1012): NLLB-zs 29.17/0.84/5.64 | NLLB-ft 33.50/0.87/4.80 | GT 42.30/0.90/3.31 | Gemini 39.49/0.90/3.08.
- KEY: metric disagreement (surface->GT #1, learned[COMET+MetricX]->Gemini #1) reproduces on INDEPENDENT FLORES+ refs => disagreement is a metric property, not a post-editing artifact. Added as a sentence to paper §4 bias paragraph.
- Tables regenerated; paper recompiles clean (7 pages, 0 undefined citations).

## 2026-06-10 — IFS+QE reranking booster: NEGATIVE RESULT (reported as a finding)
- HF token blocker resolved: cached CLI token was invalid; user added a fresh read token to .env + accepted Unbabel/wmt22-cometkiwi-da license. score_qe.py now loads HF_TOKEN from .env (dependency-free). scripts/fetch_cometkiwi.py robustly pulls the gated 2.26GB model.ckpt (rode out transient ConnectionResetError 10054 on the HF API).
- N-best generated on Kaggle GPU (scripts/kaggle_nbest.py + kaggle/run_nbest.py): dev 908 + test 846, beam-16. QE (COMET-Kiwi, CPU comet venv) scored all 28k candidate pairs (~40 min, chunk-resumable).
- Rerank ablation on test (n=846, chrF++): beam1 40.94 | qe 40.90 | ifs 40.88 | fused(α=0.2) 40.92 | MBR 41.35 | oracle 46.27. Google Translate target 43.60.
- **KEY: oracle 46.27 > GT 43.60 (a GT-beating candidate exists for 93.3% of sources), but NO reference-free selector recovers it (best MBR +0.41 vs beam1).** Diagnosis: COMET-Kiwi per-source ranking vs reference-chrF++ Spearman ρ=0.10 (dev 0.08), QE/oracle agreement 9.3% (chance 6.2%); IFS near-constant across paraphrase beam. = COMET miscalibration for Burmese made concrete at decoding time.
- Decision (user): report as honest negative finding + flagship bridge, NO parity-with-GT claim. New paper section "Can Decoding-Time Reranking Close the Gap?" + tables/rerank.tex (rerank_table() in make_tables.py reads rerank_report.json). Added cites: rei2022cometkiwi, eikema2020mbr, freitag2022mbr. Paper compiles clean, 9 pages, 0 undefined.
- New/changed code: src/rerank/rerank.py (added mbr_pick, _spearman, diagnostics()), src/rerank/score_qe.py (load_env_token), scripts/kaggle_nbest.py, scripts/kaggle/run_nbest.py, scripts/fetch_cometkiwi.py, src/eval/make_tables.py (rerank_table).
