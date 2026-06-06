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
