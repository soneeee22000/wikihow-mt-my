# Flagship RQ4 Protocol — IFS-Localized Selective Repair ("When Metrics Fail")

_Created 2026-06-12. The ACL/EMNLP 2027 flagship experiment design, and how the WMT 2026 human
study is instrumented to double as its data engine. Spine: [[BRAINSTORM]],
`docs/strategy-novelty-and-deployment.md` §10. Builds on `docs/ifs-human-validation-protocol.md`._

## 0. One-line thesis

When the quality metric is unreliable (the default for low-resource languages), scalar-reward
quality control fails — proven at decoding time in the WMT paper (reranking recovers +0.41 of a +5.3
oracle headroom; QE/ref ρ=0.10). The alternative is **structured error localization → selective,
constraint-targeted repair**: use IFS not to _rank_ candidates but to _localize_ which procedural
element broke, then regenerate only that, only where flagged.

## 1. Research question & hypotheses

**RQ4:** Does IFS-as-in-loop _localizer_ in a translate→localize→repair agent improve human
**instruction-followability** over (a) the base translator and (b) generic free-text-critique
refinement, for low-resource procedural en→my?

- **H4a:** C2 (IFS-localized repair) > C0 (base) on human followability. _(repair helps at all)_
- **H4b (the crux):** C2 > C1 (free-text-critique repair) on human followability. _(structure is
  what helps, not just "an LLM took a second look")_
- **H4c (cost/deployment):** C2 reaches ≥ X% of C3 (full-LLM retranslation) followability at a
  fraction of C3's cost/latency, by firing the LLM on only the IFS-flagged segments.

**Principled reason H4b should hold (the intellectual core):** a free-text LLM critic is _itself_
miscalibrated on Burmese fluency — same low-resource failure that breaks COMET. Structured
constraints (this numeral, this entity, this imperative, this step order) are **language-agnostic,
checkable surface anchors**: verifying them is cross-lingual _matching_, not Burmese _understanding_.
If H4b fails (C1 ≈ C2), structure added nothing — report it honestly and the paper becomes a
cost-efficiency result, not a method result.

## 2. Systems under comparison (the decisive table)

All en→my, on the **same source sentences** as the WMT human study (paired; see §4).

| ID     | System                         | What it is                                                                                                            |
| ------ | ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **C0** | Fine-tuned NLLB-600M           | base translator (already rated in the WMT study — reuse those ratings)                                                |
| **C1** | C0 + free-text-critique repair | LLM gets "improve faithfulness" free-text critique (TEaR/Ng-style ceiling for "an LLM looked again")                  |
| **C2** | C0 + **IFS-localized repair**  | IFS flags the broken component(s); LLM regenerates _only the flagged segment_ under an explicit structured constraint |
| **C3** | Full-LLM retranslation         | upper-cost reference (e.g. Gemini 2.5) translating from scratch — the "why not just use the LLM" control              |

Repair fires **only on IFS-flagged segments** for C1/C2 (selective). Log per segment: flagged?
(bool), flagged component(s), repair attempted (text), so analysis can compute **repair success rate
on the flagged component** (needs the binary component checks, §3).

## 3. What humans rate (extends the WMT instrument)

Same blind 1–5 protocol as `ifs-human-validation-protocol.md` §2, plus the **binary component
checks now made mandatory** (they were "optional/recommended"):

- **Instruction-Followability** (1–5, primary) — unchanged.
- **Adequacy** (1–5) — unchanged.
- **Component checks (Y/N)** — step-order preserved, action/verb correct, entities correct,
  quantities/units correct. **The action check is the only human-only IFS component** (WMT needs it),
  and all four let the flagship measure _whether a repair fixed the specific flagged element_.

Rationale for adding these to the WMT study NOW (before any data is collected): they strengthen the
WMT paper (human grounding for the IFS action component + per-component human validation) _and_ are
the flagship's targeting signal. One config change, two papers. (Config: `ratings_ls_config.xml`.)

## 4. Forward-compatibility rules (one study, two papers)

1. **Same 40 source sentences** (seed 42 sample from `test`) for both papers. C1/C2/C3 are generated
   on those same 40 sources → directly paired against the WMT C0 ratings; **no re-baselining**.
2. **Same raters, same onboarding, same QC, same scale.** Raters calibrated in the WMT round rate the
   flagship's repair conditions later → comparable, and reliability (Krippendorff α, test–retest)
   carries over.
3. **Blind + shuffled** across _all_ conditions (C0–C3) so raters can't tell base from repaired from
   LLM. Extend `ratings_key.csv` with the new condition labels; never upload the key.
4. **Incremental rows:** WMT = 40×4 systems = 160 rows. Flagship adds 40×3 new conditions (C1/C2/C3)
   = 120 rows, rated by the same panel. C0 is shared (already rated).

## 5. Analysis

- **Primary:** segment-level human followability, C2 vs C1 vs C0 — paired bootstrap 95% CIs (1000×),
  Wilcoxon signed-rank (paired, ordinal). System-level means reported but not the headline (small n;
  Mathur 2020).
- **Mechanism:** on segments IFS flagged, **repair-success rate** = fraction where the flagged
  component's human Y/N flips N→Y after C2 vs after C1. This is the direct evidence that _structured
  localization_ (not just "a second pass") drove the gain.
- **Cost/latency:** mean LLM calls per source and wall-clock for C1/C2/C3; C2 should be a fraction of
  C3 because it fires selectively.
- **Metric cross-check:** IFS / COMET / MetricX on all conditions (automatic), to show the
  scalar-metric blind spot persists where humans see the gain.

## 6. Cut-lines (honest, pre-registered)

- **Gate 0 (gating dependency):** RQ3 must pass first — IFS must validate against human followability
  in the WMT study (≥ COMET on the step+action sub-judgment, or an honest mixed result). If IFS ≤ BLEU
  on correlation, **both** papers fall back to resource+benchmark, and RQ4 does not run.
- **H4a fails** (C2 ≯ C0): repair doesn't help → kill the agent thesis; report negative.
- **H4b fails** (C1 ≈ C2): structure didn't matter → reframe as a cost-efficiency / selective-repair
  systems result, not a method contribution.
- **H4c only** (C3 dominates at acceptable cost): honest "just use the LLM" finding; pivot to the
  on-device/offline deployment angle.

## 7. What is NOT built yet (do not start until RQ3 confirms — user decision 2026-06-12)

The agent (localize→constrain→regenerate loop), C1/C2/C3 generation, and the generalization probe
(a 2nd domain or pair to support the "Executable Translation" breadth claim) are **designed, not
implemented**. This doc + the instrument changes are the non-premature part. Build order when greenlit:
C2 localizer/repair on the saved checkpoint + Gemini (smoke-first, few segments) → C1/C3 → generate on
the 40 sources → second rating round.
