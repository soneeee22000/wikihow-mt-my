# IFS Construct-Novelty Literature Search (adversarial)

Run 2026-06-13 by a background research agent, scope = "is IFS novel?" Adversarial brief: find
prior art that _kills_ the claim. Verdict: no work occupies the exact cell (reference-free,
source-anchored, step/action/entity/quantity decomposition, validated vs human followability,
low-resource Burmese), but the wording must be narrowed and several works cited.

> **Verification status (IMPORTANT before citing):** the agent verified pre-2026 papers directly
> via arXiv/ACL Anthology. The 2026 arXiv IDs (`2601.*`, `2602.*`, `2605.28218` IFMTBench, etc.)
> were seen only as search-result listings — **pull each by arXiv ID and confirm title/authors/
> venue before adding to `references.bib`.** Do not cite unverified IDs (no-fabrication rule).

## The biggest threat — recipe GENERATION metrics (does NOT kill us)

The exact 4-part decomposition (ingredient/entity, action-verb, quantity, step/action edit
distance) already exists in monolingual **recipe/procedural generation** eval — Bosselut et al.
"Simulating Action Dynamics with Neural Process Networks" (ICLR 2018, arXiv:1711.05313, VF1/SF1);
RecipeGen; "Losses that Cook" (arXiv:2601.02531, UNVERIFIED). Components are therefore **adopted,
not invented** — every such paper scores a _generated_ recipe against a _same-language reference_,
none is cross-lingual / reference-free / followability-validated. => must say we _adapt_ these
signals into reference-free MT eval.

Correction: the TACL 2024 "Cultural Adaptation of Recipes" (Cao et al., arXiv:2310.17353) uses
**only** BLEU/chrF/ROUGE-L/BERTScore — no decomposition. Adjacent (recipe translation) but not a
threat.

## By threat angle

- **Numeral/entity preservation in MT (our "adopted" components):** Wang, Xu, Guzmán, El-Kishky,
  Rubinstein, Cohn, "As Easy as 1, 2, 3: Behavioural Testing of NMT Systems for Numerical
  Translation," Findings of ACL 2021 (2021.findings-acl.415) — reference-free numeral testing,
  strong precedent for our QUANTITY component. KoBE (arXiv:2009.11027) — entity-based reference-free
  MT scoring. SynCED-EnDe 2025 (arXiv:2510.05144, UNVERIFIED) — newer critical-error dataset.
- **Interpretable/decomposed MT metrics:** xCOMET (Guerreiro et al., TACL 2024, learned, MQM-typed
  — the kind we argue is miscalibrated for Burmese); Park, "Multi-Dimensional MT Evaluation … for
  Korean," LREC-COLING 2024 (arXiv:2403.12666) — decomposed **reference-free** MQM dimensions =>
  **do not claim "first decomposed reference-free MT metric"**; our novelty is the _procedural_ axis.
- **Task-based / functional / followability eval (our core angle):** White & Taylor, "Task-based
  Evaluation for MT," MT Summit VII 1999 — conceptual ancestor (cite so reviewers see we know it).
  İlgen & Hattab, "Toward Human-Centered Readability Evaluation," arXiv:2510.10801 (2025) — names an
  "actionability" dimension but is **monolingual readability and explicitly unvalidated**.
- **LLM instruction-following (our stated distinction):** IFEval (2311.07911), XIFBench (2503.07539)
  — already cited. **IFMTBench, "A Comprehensive Benchmark for Multilingual Translation Instruction
  Following," arXiv:2605.28218 (2026, UNVERIFIED)** — bridges IFEval-style constraints into MT; the
  most confusable recent work. If verified, cite and draw the line: IFS = followability of the
  _procedure_ by an end user, NOT the MT system's compliance with author constraints.
- **Low-resource / Burmese / SEA:** Turkish wikiHow procedural benchmark (arXiv:2309.06698) —
  procedural + low-resource + wikiHow but standard metrics, no procedural-fidelity metric. "When
  LLMs Struggle: Reference-less Translation Evaluation for Low-resource Languages" (arXiv:2501.04473).
  **No prior Burmese MT metric-vs-human-followability study found => "first for Burmese" is clean.**

## Verdict + recommended narrowed headline

Defensible with two narrowings:

1. step+action are **not novel signals** (standard in recipe generation) — novel is _importing them
   into reference-free MT evaluation_.
2. task-based/followability eval is **not a brand-new concept** (White & Taylor 1999) — novel is an
   _automatic, decomposed, reference-free metric whose components are validated against human
   followability_.

Recommended honest headline:

> IFS is, to our knowledge, the first reference-free, source-anchored MT metric that decomposes
> procedural fidelity into step, action, entity, and quantity preservation and validates these
> against human instruction-followability ratings — adapting structured-fidelity signals previously
> used only in monolingual procedural-text generation, and providing the first such validation for a
> low-resource language (Burmese).

Do NOT claim "first interpretable/decomposed reference-free MT metric" (Park 2024, xCOMET, KoBE).

## Candidate citations to add (verify 2026 IDs first)

Verified pre-2026: Wang et al. 2021 (numeral behavioural testing); Park 2024 (2403.12666); White &
Taylor 1999; Bosselut et al. 2018 (1711.05313); İlgen & Hattab 2025 (2510.10801); Cao et al. 2024
(2310.17353); Turkish wikiHow 2023 (2309.06698); KoBE (2009.11027). xCOMET TACL 2024 already in bib.
Verify before citing: IFMTBench (2605.28218), Losses that Cook (2601.02531), SynCED-EnDe (2510.05144).
