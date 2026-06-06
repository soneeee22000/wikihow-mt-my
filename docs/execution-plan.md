# WMT 2026 Paper — Execution Plan v2 (Research-Informed Revision)

**Supersedes:** `WMT2026_Execution_Plan.md` (the original handoff). That plan is sound; this v2 patches it with verified external research (4 parallel research passes, 2026-06-06) and with what is **actually on disk** in this workspace.
**Project:** WikiHow English→Myanmar Low-Resource Instructional MT
**Venue:** WMT 2026 research track (EMNLP 2026, Budapest, 28–29 Oct 2026)
**Owner:** Pyae Sone (sole first author)
**Revised:** 2026-06-06

---

## 0. What changed vs v1, and why (read this first)

Every item below is backed by a verified source or by direct inspection of the files in `Thesis Papers/DATASET-CREATION/`.

| #   | Change from v1                                                                                                                                                                                               | Reason                                                                                             | Source                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ---------------------- |
| C1  | **Dataset is "human post-edited MT (MTPE)," not "gold-standard human translation."** Reframe everywhere.                                                                                                     | Folder is literally `WikiHow-MTPE`; a separate `human_written_data_5K.csv` exists.                 | On-disk inspection     |
| C2  | **Primary metric = chrF++ (not BLEU).** Report spBLEU for FLORES-comparability, COMET for semantics, and add **MetricX-24 or COMET-Kiwi**.                                                                   | Burmese is unsegmented → BLEU unreliable; FLORES/NLLB use chrF++/spBLEU.                           | NLLB / FLORES-200      |
| C3  | **Must cite + cross-compare BURMESE-SAN (arXiv 2602.18788, Feb 2026).**                                                                                                                                      | Closest neighbor; benchmarks NLLB+LLMs on Burmese with MetricX. A reviewer _will_ ask.             | Research pass 4        |
| C4  | **IFS repositioned:** headline = _ordered-step + imperative-action preservation_ as a downstream **followability** metric; entity + quantity are _adopted_ signals (cite M-ETA, WMT critical-error NUM/NAM). | IFS as a 4-way weighted sum is only combination-novel; step+action are the only unclaimed parts.   | Research pass 3        |
| C5  | **Dataset release = rehydration pattern** (MY translations + WikiHow URLs/IDs + alignment + a `rehydrate.py`), licensed **CC BY-NC-SA 3.0**. Never mirror the English text.                                  | WikiHow is CC BY-NC-SA 3.0; ShareAlike forces same license; this is how WikiLingua/Koupaee did it. | Research pass 2        |
| C6  | **Deadline is "August 2026," exact day NOT yet posted.** Plan to ~Aug 14 (WMT25 proxy), re-anchor when statmt.org/wmt26 posts.                                                                               | Official site shows month only.                                                                    | Research pass 1        |
| C7  | **Solo-rater is the #1 risk to IFS.** Mitigate: recruit 2–3 light raters (SpeakProof network) for a ~50-segment overlap subset + author test-retest.                                                         | n=1 human ground truth is the soft spot a metrics reviewer attacks.                                | Decision, this session |
| C8  | **Phase 1 is ~70% done but not finished.** Canonical-version selection + dedup + normalization + split-integrity check required.                                                                             | Duplicate rows, mixed numerals, whitespace, schema sprawl seen on disk.                            | On-disk inspection     |

**Decisions locked this session:**

- Direction: **en→my only** (scope discipline; my→en parked for follow-up).
- IFS: **build it, repositioned**, with full human-correlation validation (C4 + C7).
- LLM few-shot baseline: **Gemini 2.5** (strongest public system on Burmese per FLORES).
- Annotators: **author + recruit 2–3 light raters**; fall back to author + test-retest if recruiting fails.

---

## 1. Verified target & constraints (re-anchor the calendar)

| Item                        | Value                                                                                                                                                                                                                                 | Confidence                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Venue                       | WMT 2026 research track, co-located EMNLP 2026                                                                                                                                                                                        | VERIFIED                    |
| Conference                  | **28–29 Oct 2026, Budapest**                                                                                                                                                                                                          | VERIFIED                    |
| Submission route            | Direct via SoftConf `softconf.com/emnlp2026/wmt2026/` (ARR commitment also allowed, **not required**)                                                                                                                                 | VERIFIED                    |
| Paper deadline              | **"August 2026" — exact day NOT posted.** Plan to ~Aug 14 AoE (WMT25 proxy).                                                                                                                                                          | day UNCERTAIN               |
| Notification / camera-ready | September 2026 (month only)                                                                                                                                                                                                           | day UNCERTAIN               |
| Length / format             | 6–10 pp, anonymized, EMNLP style, Limitations excluded from count                                                                                                                                                                     | VERIFIED                    |
| Acceptance reality          | **Competitive peer review (~45–60%), NOT auto-accept.** ~60 research submissions in 2025.                                                                                                                                             | VERIFIED (rate approximate) |
| Topic fit                   | "selection and preparation of data for MT," "multilingual MT," "automatic methods for evaluating MT, quality estimation" all explicitly in CFP. "Low-resource" not verbatim — frame via "multilingual / transfer / data preparation." | VERIFIED                    |

**Action P0 (Week 1):** confirm exact deadline from `www2.statmt.org/wmt26/index.html`; update the timeline in §11.

---

## 2. Thesis & contributions (final framing)

**Working title:** _"WikiHow-MY: A Human Post-Edited English–Myanmar Instructional MT Corpus and a Step-and-Action Followability Metric."_

1. **The corpus (headline).** ~10k English–Myanmar WikiHow instructional pairs, **human post-edited (MTPE) and quality-checked**, cleaned, article-disjoint splits, documented with a datasheet, released under CC BY-NC-SA 3.0 via rehydration. First instructional-domain en–my resource (the niche is genuinely open — ALT/UCSY/FLORES are all news/general; WikiLingua excludes Burmese).
2. **The benchmark.** NLLB-200-distilled-600M (zero-shot + fine-tuned) vs **Gemini 2.5 few-shot** vs Google Translate, scored chrF++ / spBLEU / COMET / **MetricX-24**, with a **FLORES+ cross-eval row** to quantify the news→instructional domain gap and anchor against BURMESE-SAN.
3. **IFS — a step-and-action followability metric.** Reference-based. Headline components = **ordered-step preservation** and **imperative-action preservation** (no existing MT metric scores these); entity and quantity are **adopted** critical-error signals (M-ETA; WMT-CED NUM/NAM). Validated by correlation with **human instruction-followability ratings** vs BLEU/chrF/COMET, with significance testing (bootstrap / William's test vs COMET).

**Honesty guardrails (non-negotiable):**

- Never claim "gold-standard human translation" — say "human post-edited / MTPE, quality-checked."
- Never claim "first metric for entity/number preservation" or "first instruction-faithfulness metric" — both false. Claim: _first reference-based procedural-MT followability metric combining step+action (novel) with entity+quantity (adopted), validated against human followability in a low-resource language._

**Cut-line (unchanged):** if by end of Week 5 IFS doesn't beat BLEU on human correlation, report it honestly as a mixed result OR fall back to a dataset+benchmark resource paper (contrib 1–2). A smaller submitted paper beats a bigger missed one.

---

## 3. Related Work landmines to pre-empt (cite each head-on)

- **M-ETA / XC-Translate (EMNLP 2024)** — reference-based entity-translation accuracy. _Your entity component already exists._ Cite as adopted signal.
- **RecipeGen (2025)** — step/ingredient/faithfulness metrics for recipe _generation_. _Closest to your step triad._ Differentiate: yours is cross-lingual MT, not monolingual generation.
- **XIFBench (NeurIPS 2025)** — instruction-following + numerical constraints + preservation + low-resource. _Framing collision._ Differentiate: IFS is a deterministic reference-based MT metric (no LLM judge needed — critical where no strong Burmese LLM judge exists), not a benchmark.
- **WMT Critical Error Detection (NUM/NAM)** — number/entity preservation in MT is a recognized, pre-defined problem. Cite as the lineage your entity/quantity components inherit.
- **BURMESE-SAN (2026)** — the en–my incumbent. Cite, cross-eval on FLORES+, and add MetricX so you can't be dismissed on metric choice.
- **METEOR / MQM** — establish that "weighted sub-scores tuned to humans" is a 20-year-old pattern, so the _construct_ (followability) is the contribution, not the formula.

---

## 4. Phase 1 — Data (Week 1) — REWRITTEN to match disk

Current state: many overlapping files in `Thesis Papers/DATASET-CREATION/` and `Thesis Papers/01 Translate!/` (`en_mm_parallel_corpus.csv`, `Data10KFinal.csv`, `enmy10K.csv`, `6Kdata.csv`, `human_written_data_5K.csv`, per-article `WikiHow-MTPE/*.xlsx`, `master_spreadsheet.csv`, existing `train/test/valid.csv`). Schemas differ (`English,Burmese` / `my,en` / `Source text,Target text`).

Steps:

1. **Provenance audit + canonical selection.** Determine exactly how each file was made (which are MTPE, which "human_written," which are duplicates/older versions). Pick ONE canonical 10k. Write `src/data/provenance.md` documenting source-by-source: count, creation method (MTPE vs human-written), and which WikiHow articles. This doc becomes the paper's datasheet backbone.
2. **Unify schema** (`load.py`): all → `{id, article_id, article_title, category, en, my, source_url, provenance}` (provenance ∈ {mtpe, human_written}). Keep `article_id` — needed for disjoint splits.
3. **Myanmar normalization** (`normalize_my.py`): Zawgyi→Unicode via `myanmar-tools` (log Zawgyi row count — expect ~0 but verify), NFC, collapse spurious internal whitespace, normalize punctuation (၊ ။), and **normalize numerals** (Myanmar ၀–၉ ↔ Arabic) — decide one canonical form, log the rule (this rule is reused by IFS quantity).
4. **Clean** (`clean.py`): drop exact + near-duplicate rows (the `"Step 1 …"` repeats are real), strip residual list/step numbering artifacts into a structured `step_index` field rather than inline text, length-filter, flag misaligned rows.
5. **Quality spot-check:** sample 100 pairs → CSV; rate alignment/quality; **log the error rate** (goes in the dataset-quality paragraph). Since you are a native speaker, do this yourself.
6. **Split integrity** (`split.py`): **verify the existing train/test/valid are article-disjoint and seeded.** If they are sentence-level random splits → **rebuild** stratified-by-category, **80/10/10, split by `article_id`, seed 42.** Article leakage invalidates the whole benchmark — this is the single most important Phase-1 check.
7. **Stats** (`stats.py`): counts, pairs/article, EN vs MY length distributions, category distribution, MTPE vs human-written split. Emit `dataset_stats.json` + figures.

**Deliverables:** `data/processed/{train,dev,test}.jsonl` (article-disjoint, seeded); `provenance.md`; `dataset_stats.json`; 1 distribution figure; spot-check error rate logged.

---

## 5. Phase 2 — Baselines (Weeks 2–3)

Systems (en→my): NLLB-200-distilled-600M zero-shot; NLLB fine-tuned (the workhorse); **Gemini 2.5 5-shot**; Google Translate.
**Anchors from research:** NLLB en→my chrF++ ≈ 34 on FLORES (verify against NLLB appendix before quoting); expect distilled-600M zero-shot a few points lower; expect into-Myanmar harder than into-English. Domain shift (FLORES news → WikiHow) means don't expect an exact match.
Fine-tune config: as v1 §6 (LR 3e-5, early-stop on **chrF**, seed 42, beam 5) — that config is good, keep it.
**Add:** a cross-eval row — fine-tuned model on FLORES+ Myanmar test split — to connect to BURMESE-SAN's general-domain numbers and quantify the domain gap.

**Deliverables:** checkpoint (Drive/Hub), hypotheses for all systems, `main_results.json`, `experiments/LOG.md` with exact commands + sacreBLEU signature.

---

## 6. Phase 3 — IFS (Weeks 3–5) — repositioned

Components (`src/eval/ifs.py`): **step** (LCS/Kendall-tau over step boundaries — use the `step_index` from Phase 1), **action** (imperative-verb recall via reference alignment), **entity** (F1 vs bilingual term list — adopted, cite M-ETA), **quantity** (exact match after numeral normalization — adopted, cite WMT-CED NUM). Start weights α=.4 β=.3 γ=.2 δ=.1, then **tune on dev to maximize correlation with human followability** and report that.
**Validation = the contribution:** segment-level + system-level Pearson/Spearman of IFS vs human followability, against BLEU/chrF/COMET, **with significance testing**. Be honest if COMET wins overall — IFS can still win on the instruction-specific sub-judgment.

**Deliverables:** `ifs.py` + unit tests on toy examples; `correlate.py` correlation table; tuned weights logged.

---

## 7. Phase 4 — Human eval + cultural analysis (Weeks 4–5)

- **You rate** ~150–200 segments on Adequacy + Instruction-Followability (1–5).
- **Recruit 2–3 light raters** (SpeakProof learners / Ekkhara / family) for a **~50-segment overlap subset** → one Krippendorff's α. This single number converts "author opinion" into a validated metric.
- **Author test-retest:** re-rate 50 segments ≥2 weeks apart → intra-rater reliability.
- Fallback if recruiting fails: solo + test-retest, framed as the headline limitation.
- **Cultural analysis (~0.5pp):** 30–50 examples where literal MT is culturally off; table of source / NLLB output / culturally-adapted target. Frame as motivation for future CEA work.

**Deliverables:** ratings CSV; α (or test-retest) number; error-taxonomy counts; cultural-examples table.

---

## 8. Eval harness, repo, writing — as v1 §3/§9/§10, with:

- `automatic.py` outputs chrF++ (primary), spBLEU, COMET, **MetricX-24**, and calls `ifs.py`. All metrics → JSON keyed by system; `make_tables.py` generates LaTeX (no hand-typed numbers).
- Drafting order: Dataset → Experiments → IFS → Analysis → Related Work → Intro → Abstract → Limitations.
- Limitations must name: ~10k scale, single pair, **MTPE provenance**, small/solo human eval, NC license limits downstream commercial use.

---

## 9. Release checklist (data)

- [ ] Release MY translations + WikiHow URL/article-ID + alignment + `rehydrate.py`; **do not** mirror English text.
- [ ] License data **CC BY-NC-SA 3.0**; code separately (MIT/Apache). Tag HF card `cc-by-nc-sa-3.0` correctly.
- [ ] Datasheet: provenance (MTPE), translator/post-editor qualifications + compensation, Burmese encoding (Unicode), intended use (research, NC).
- [ ] License-provenance paragraph in the paper.
- [ ] Verify WikiHow footer CC version (3.0 vs 4.0) before camera-ready.

---

## 10. Re-anchored timeline (to ~Aug 14)

| Week | Dates 2026   | Focus                                                                                   | Exit criterion                               |
| ---- | ------------ | --------------------------------------------------------------------------------------- | -------------------------------------------- |
| 1    | Jun 8–14     | Phase 1: provenance audit, canonical 10k, normalize, **verify article-disjoint splits** | clean seeded article-disjoint splits + stats |
| 2    | Jun 15–21    | NLLB zero-shot + fine-tune (Colab)                                                      | fine-tuned beats zero-shot on dev chrF       |
| 3    | Jun 22–28    | Gemini + GTrans + FLORES+ cross-eval; IFS skeleton                                      | main results v1; IFS runs                    |
| 4    | Jun 29–Jul 5 | IFS components; launch your ratings + recruit raters                                    | IFS scores; ratings underway                 |
| 5    | Jul 6–12     | IFS validation + cultural analysis; **go/no-go on contrib 3**                           | correlation table; scope locked              |
| 6    | Jul 13–19    | Draft Dataset + Experiments + IFS                                                       | half paper in LaTeX                          |
| 7    | Jul 20–26    | Remaining sections; all tables/figures final                                            | full first draft                             |
| 8    | Jul 27–Aug 2 | Review, polish, anonymize, format                                                       | submission-ready                             |
| —    | Aug 3–14     | Buffer + re-runs                                                                        | **submit**                                   |

---

## 11. Open items needing your input before Phase 1 finishes

1. **Provenance reality:** how was the 10k made — fully MTPE, or 5K human-written + 5K MTPE? (Determines the datasheet + honest framing.)
2. **Existing splits:** were `train/test/valid.csv` split by article or by sentence? (If sentence → I rebuild.)
3. **MetricX-24:** OK to add (needs a model download + a little compute) — it's our defense against BURMESE-SAN's metric critique?
4. **Rater recruiting:** can SpeakProof realistically yield 2–3 people for ~50 segments by Week 4?

---

_v2 ends. Begin at §4 Phase 1 once items 1–2 above are answered. Append every run to `experiments/LOG.md`._
