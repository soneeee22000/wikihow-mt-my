# Literature Review & Positioning — WikiHow en→my MT (two-paper arc)

_Synthesis of a 5-area cited review (2026-06-08). This is the scholarly spine: Related Work
foundation, the honest novelty verdict, research questions/hypotheses, corrected contribution
claims, the defensive eval matrix, and verification gaps to close before camera-ready._

---

## A. Low-resource & English–Burmese MT

- **NLLB-200** (Costa-jussà et al., Nature 2024; arXiv 2207.04672) — the must-have open NMT baseline; reports spBLEU/chrF++ on FLORES-200. (Pull exact en→my numbers from the FLORES leaderboard for camera-ready.)
- **FLORES-200 / FLORES+** (Goyal et al. TACL 2022; OLDI 2024) — canonical eval set; `mya_Mymr` included. → **add a FLORES+ cross-eval row.**
- **Asian Language Treebank (ALT)** (Thu et al., LREC 2016) — the de-facto prior en–my parallel set, but **news domain** (~18k/1k/1k). Our key domain contrast.
- **WikiLingua** (Ladhak et al., Findings EMNLP 2020) — WikiHow-derived, 18 languages, **Burmese NOT included** (verified vs repo). Same source lineage, no Burmese → our cleanest novelty contrast.
- **BURMESE-SAN** (arXiv 2602.18788, 2026) — newest Burmese LLM benchmark; MT subtask = FLORES+, scored with **MetricX-24**; Gemini-class leads. → **add MetricX-24** for comparability.
- Others to cite: M2M-100, mBART-50 (classic baselines), myXNLI (Burmese low-resource context), UCSY (general en–my), MyanmarGPT.
- **Burmese challenges (cite):** unsegmented script (no word boundaries) → char-level metrics; Zawgyi↔Unicode (we normalized to Unicode — a methods point); chrF++/spBLEU over BLEU.

## B. MT evaluation metrics & faithfulness (positioning IFS)

- Surface: **BLEU** (Papineni 2002), **chrF/chrF++** (Popović 2015/2017), **spBLEU** (FLORES). Learned: **COMET** (Rei 2020), **COMET-Kiwi/QE** (2022), **BLEURT** (2020), **MetricX-24** (Juraska, WMT 2024).
- **COMET is miscalibrated for low-resource — documented:** Falcão et al. (LREC-COLING 2024), AfriCOMET (NAACL 2024), SSA-COMET (EMNLP 2025). → our core "don't just use COMET for Burmese" evidence.
- **Entity + quantity preservation are PRECEDENTED — must cite, do NOT claim novel:** WMT Critical Error Detection **NUM/NAM** (WMT'21/22 QE Task 3); **M-ETA / XC-Translate** (Conia et al., EMNLP 2024, entity-translation accuracy); **ACES** (number/entity challenge sets, WMT 2022); HalOmi (omission/hallucination, EMNLP 2023).
- **RecipeGen** (arXiv 2506.06733, 2025) — ⚠️ **the closest pre-empt:** Action Precision, step edit-distance (ordering), ingredient recall (entity), quantity precision — ~3 of our 4 components, **but for recipe _generation_, not MT, and not followability-validated.** Cite prominently + differentiate.
- **METEOR** (Banerjee & Lavie 2005) — already rewards _intra-sentence word_ order; distinguish from our _procedural step_ order.
- Name check: **no MT metric named "Instruction Faithfulness Score"** (2308.12674 is a training method, not a metric — no clash). Add a footnote distinguishing IFS from RAG/QA "faithfulness."

## C. Procedural text & "instruction-followability" (the construct)

- **We can claim/operationalize the construct** — it is _not_ an established named term. Frame as the intersection of:
  - **Task-based MT evaluation** (GALE tradition; Voss & Tate) — can a user complete a task from MT output? (the "how")
  - **Skopos / functionalist adequacy** (Reiss & Vermeer 1984; instructions = "operative" texts judged by whether they elicit correct reader action) — (the "why").
- **Disambiguate hard from IFEval** (Zhou 2023) and **XIFBench** (NeurIPS 2025) — those = an _LLM obeying a prompt_, NOT _a translated how-to remaining executable_. Reviewers WILL conflate; state the difference in Intro + Related Work.
- Step/action structure to cite: SRL for procedures (2505.21068), imperative-verb extraction (1806.07999), MQM critical-error severity = physical harm (Freitag TACL 2021) → why step/quantity errors are _critical_ in procedural text.
- Closest prior: GALE task-based eval + RecipeNLG (procedural translation as a task) — neither asks "can the human still perform the procedure?" on a low-resource pair.

## D. Human-evaluation methodology (reviewer-proof design)

- **Scales:** Direct Assessment (Graham 2013) — continuous/Likert, **z-normalize per rater**; MQM (Freitag TACL 2021) rationale for a small _expert native_ panel over anonymous crowd. Tool: **Appraise** (Federmann 2018) or a faithful clone (≈ our Label Studio setup).
- **Correlation reporting (the plan):** **segment-level Spearman (primary, outlier-robust) + Pearson, each with 95% bootstrap CIs (1000×), n≈160.** Do **not** report system-level (n too small — Mathur "Tangled up in BLEU", ACL 2020).
- **Significance:** **Williams (Hotelling–Williams) test** for IFS-vs-COMET / IFS-vs-chrF dependent correlations (Graham & Baldwin, EMNLP 2014) — already in `correlate.py`. Bootstrap CIs per Kocmi "To Ship or Not to Ship" (WMT 2021).
- **Agreement:** **Krippendorff's ORDINAL α** (not κ) on a **≥50-item (~30%) overlap subset**; **intra-rater test–retest** (~20 items, small panel); **~20% QC items** (exact repeats + degraded "bad" outputs) with documented rater filtering; native-Burmese rater profiles (L1, English level) documented (FLORES rater norms).
- Minimum defensible design: all raters cover the ≥50 overlap subset + QC; remaining items single-coded to cover all 160; lead with Spearman + Williams + CIs.

## E. Agentic/LLM MT & metric-in-the-loop (Track-2 flagship)

**White-space test PASSES** — no work uses a structured procedural-fidelity score (step+action+entity+quantity) as an **in-loop reward/termination signal** in an MT agent, none for low-resource procedural text.

- Already done (do NOT claim): reflection/self-refine MT (**Self-Refine** NeurIPS 2023; **Ng translation-agent** 2024); MQM/MetricX-driven refinement (**TEaR** NAACL-F 2025 — _closest in spirit_; **LLMRefine** NAACL-F 2024); COMET/xCOMET as RL reward (**MT-R1-Zero**, **MT-RewardTree** EMNLP-F 2025; **TAT-R1** word-alignment reward 2025); QE reranking (**MAPS** TACL 2024, QUEST); multi-agent (**TransAgents** TACL 2025, **DelTA** ICLR 2025); faithfulness via _training_ augmentation (**SWIE/OVERMISS** 2308.12674).
- Differentiation pivots: (1) vs TEaR/LLMRefine/Ng — their in-loop signal is generic quality (MQM/MetricX/free-text); ours is a **structured procedural rubric that gates termination**. (2) vs MT-R1/RewardTree/TAT-R1 — their reward is COMET-family/terminology at _training_ time; ours is procedural fidelity as an _inference-time agent control signal_. (3) vs COMET-reward's own documented pitfalls/reward-hacking — ours is interpretable + reference-light by construction.

---

## Novelty verdict — what we CAN and CANNOT claim

| Claim                                                                                         | Verdict                                                                                                         |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| First **instructional/how-to en–my** parallel corpus                                          | ✅ holds (WikiLingua has no Burmese; ALT is news). Phrase precisely.                                            |
| IFS **entity / quantity** components                                                          | ❌ NOT novel — cite WMT-CED NUM/NAM, M-ETA, ACES, RecipeGen.                                                    |
| IFS **step-ordering / action-verb** as an **MT** metric                                       | ◐ "first in MT, **to our knowledge**" — precedented in procedural _generation_ (RecipeGen, SOP); cite + credit. |
| IFS validated against human **instruction-followability** (low-resource, COMET miscalibrated) | ✅ cleanest standalone novelty.                                                                                 |
| "Instruction-followability" as a named construct (task-based × skopos)                        | ✅ can name/operationalize; disambiguate from IFEval.                                                           |
| Flagship: structured procedural-fidelity score as **in-loop reward** in an MT agent           | ✅ unclaimed (Track 2).                                                                                         |

## Research questions & hypotheses

- **RQ1 (resource):** Does an instructional en→my corpus + fine-tuning beat zero-shot/general-domain? **H1: yes** — confirmed (FT +5.63 chrF++ over zero-shot).
- **RQ2 (benchmark):** How do open NMT / commercial / LLM systems compare on instructional en→my, and do surface vs semantic metrics agree? **H2: metrics disagree** — confirmed (Google Translate best on chrF++/spBLEU; Gemini best on COMET).
- **RQ3 (metric — keystone):** Does IFS correlate with human instruction-followability **at least as well as COMET/chrF**, especially on the step+action sub-judgment? _(open — the human study; Williams test decides.)_
- **RQ4 (flagship, Track 2):** Can IFS-as-in-loop-reward in a translate→reflect→repair agent improve followability over the base translator and generic-QE-guided refinement?

## Corrected contributions (WMT 2026 paper)

1. **WikiHow-MY corpus** — first instructional/how-to en–my MTPE corpus, article-disjoint splits, datasheet, rehydration release. (cite WikiLingua, ALT)
2. **Benchmark** — NLLB(zero/FT) / Gemini 2.5 / Google Translate × chrF++/spBLEU/BLEU/COMET (+FLORES+ cross-eval +MetricX-24); finding: surface vs semantic metrics disagree on the best system.
3. **IFS** — interpretable procedural-fidelity metric for **MT** (step+action first-in-MT w/ credit to RecipeGen; entity+quantity adopted from WMT-CED/M-ETA), **validated against human followability** in low-resource Burmese where COMET is miscalibrated; construct = task-based MT eval × skopos. (cut-line: if IFS ≤ BLEU on human correlation, fall back to resource+benchmark paper.)

## Defensive eval matrix

{NLLB-200 (600M; +1.3B if feasible), Google Translate, Gemini 2.5, fine-tuned NLLB} × {WikiHow-MY test, FLORES+ devtest} × {chrF++, spBLEU, COMET, **MetricX-24**, IFS}.

## Action items emerging from the review

- **Experiments:** add FLORES+ Myanmar cross-eval row + **MetricX-24** scoring (pull NLLB FLORES baseline numbers).
- **Tooling:** `correlate.py` → add **bootstrap 95% CIs** + report **Spearman as primary**; (Williams already present).
- **Protocol/`ifs-human-validation-protocol.md`:** specify ordinal Krippendorff α, ≥50-item overlap subset, intra-rater test–retest, ~20% QC items (repeats + degraded), documented native-Burmese rater profiles.
- **Rating sheet:** seed QC/attention items (exact repeat + a degraded output that should score 1).
- **Writing:** Related Work from §A–E; Intro must disambiguate followability from IFEval/XIFBench.

## Verification gaps to close before camera-ready

1. One final **OPUS + HuggingFace Hub** search for any stray "wikihow my" parallel data.
2. Pull exact **NLLB en→my FLORES chrF++/spBLEU** from the FLORES leaderboard.
3. One negative-confirming search: "translation" + "actionability/executability" + "instructions" (construct novelty).
4. Confirm instructional-domain MT corpora are genuinely rare in _any_ pair (calibrate "rare in general" wording).
