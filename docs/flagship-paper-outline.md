# Flagship Paper Outline — "When Metrics Fail" (TACL primary / NAACL 2027 fallback)

_Created 2026-06-12. The section-by-section spine for the flagship. Each section maps to its source
material and is tagged **[WRITE NOW]** (stable, not gated) or **[GATED: RQ3/RQ4]** (needs the human
study + the C0–C3 agent). Format target: TACL (no hard page limit; ACL style). Companion resource
paper (WikiHow-MY → WAT) is tracked separately. Spine docs: [[BRAINSTORM]],
`docs/flagship-rq4-protocol.md`, `docs/strategy-novelty-and-deployment.md` §10–11,
`docs/literature-review.md` §E._

## Working title (pick at camera-ready)

- A: **"When Metrics Fail: Structured Faithfulness Localization for Repairing Low-Resource Procedural
  Translation"**
- B: "Executable Translation: Repairing Instruction-Faithfulness in Low-Resource MT When Learned
  Metrics Can't Score It"
- C: "You Can't Rerank What You Can't Rank: Structured Repair for Procedural MT in Low-Resource
  Languages"

## One-paragraph abstract (draft shell — fill numbers at RQ4)

Procedural text must stay _executable_ in translation: a dropped quantity, a reordered step, or a
wrong action makes a how-to unfollowable — a failure surface metrics miss and learned metrics, for
low-resource languages, cannot reliably score. We show on English→Burmese that the field's standard
quality-control move — selecting or rewarding translations by a scalar learned signal (COMET-Kiwi
reranking, MBR, learned-metric reward) — **provably cannot** recover available quality: a
reference-optimal selection from an open model's own beam beats a strong commercial system, yet no
reference-free scalar selector closes more than `[X]`% of the gap (QE-vs-reference ρ=`[0.10]`).
We argue the fix is not a better scalar but a **structured one**: we use an interpretable
instruction-faithfulness decomposition (step / action / entity / quantity) not to _rank_ candidates
but to _localize_ which procedural element broke, then drive **selective, constraint-targeted repair**
— regenerating only flagged segments under explicit structured constraints. Against human
instruction-followability ratings, structured-localized repair improves followability by `[Δ]` over
the base translator and by `[Δ]` over generic free-text-critique refinement, at `[k]`% of the cost of
full-LLM retranslation. `[Deployment sentence.]` `[Generality sentence.]`

---

## §1 Introduction — **[WRITE NOW]**

Source: strategy §10, BRAINSTORM "thesis spine", lit-review §C/§E.

- Hook: executable/procedural translation; the unfollowability failure mode (dropped negation, wrong
  dose, reordered step). High-stakes; worst for low-resource.
- The standard reflex (scalar quality signal: reward/rerank/MBR) and _why it fails for low-resource_
  (COMET miscalibration — cite Falcão, AfriCOMET, SSA-COMET).
- Our prior result (cite the WMT/floor paper) made this concrete at decoding time: oracle headroom
  exists, no scalar recovers it. ← the motivating negative result.
- The pivot: structured **localization → selective repair**, not a better scalar.
- Contributions list (4): (1) problem framing — executable translation + the scalar-signal failure;
  (2) the localize→repair method; (3) the principled "structured beats free-text for low-resource"
  claim + the C0–C3 human evaluation; (4) the deployment as real eval. Disambiguate "followability"
  from IFEval/XIFBench up front.

## §2 Related Work — **[WRITE NOW]**

Source: `docs/literature-review.md` §A–E (already cited + synthesized).

- Low-resource & Burmese MT (NLLB, FLORES+, ALT, BURMESE-SAN).
- MT metrics + their low-resource miscalibration (COMET, COMET-Kiwi, MetricX; Falcão/AfriCOMET/SSA).
- Agentic / reflect-refine / metric-in-the-loop MT — **the differentiation table**: TEaR, LLMRefine,
  Self-Refine, Ng, MAPS, MT-R1-Zero, TransAgents, DelTA. Our delta: structured procedural rubric as a
  **localizer driving selective repair**, not generic quality as a scalar reward/critique.
- Procedural-fidelity precedents (RecipeGen, M-ETA, WMT-CED NUM/NAM, ACES) — adopt + credit, don't
  claim. Followability construct (task-based eval × skopos).

## §3 Background: The Scalar-Signal Failure (recap + sharpen) — **[WRITE NOW]**

Source: WMT floor paper §rerank + `experiments/results/rerank_report.json`.

- Restate the reranking negative result as _this_ paper's premise (self-cite the floor paper): oracle
  > commercial, scalar selectors flat, ρ=0.10, headroom in non-top beams. This is the empirical
  > motivation; keep it tight (1/2 page) — the floor paper has the full table.

## §4 Method: IFS-Localized Selective Repair — **[WRITE NOW]** (design is fixed)

Source: `docs/flagship-rq4-protocol.md` §0–2, `src/eval/ifs.py`.

- IFS as a **structured localizer**: per segment, emit which component(s) failed (step/action/entity/
  quantity) + the specific offending source anchor.
- The repair loop: translate (NLLB) → localize (IFS) → if flagged, regenerate that segment with an
  LLM under an explicit structured constraint ("preserve quantity 350°F", "restore dropped step").
- **Selective** firing (only flagged segments) — the cost/offline argument.
- Why structured constraints beat free-text critique _for low-resource_ (the intellectual core):
  language-agnostic checkable anchors vs a critic that's itself miscalibrated on Burmese.
- Conditions C0–C3 defined here (base / free-text-critique / IFS-localized / full-LLM).

## §5 Experimental Setup — **[PARTIAL: design WRITE NOW, numbers GATED]**

Source: `docs/flagship-rq4-protocol.md` §3–4, `docs/ifs-human-validation-protocol.md`.

- Data: WikiHow-MY test; the 40 seed-42 source sentences shared with the human study.
- Systems C0–C3; base translator = the published fine-tuned NLLB; repair LLM = `[Gemini 2.5 / TBD]`.
- Human protocol: blind 1–5 followability + adequacy + 4 binary component checks; raters; Krippendorff
  α; QC. (Forward-compatible instrument already built.)
- Automatic metrics cross-check: IFS/COMET/MetricX on C0–C3.

## §6 Results — **[GATED: RQ4]** (do not write numbers until the study runs)

Source: the C0–C3 human study (not yet run).

- Main table: human followability C0/C1/C2/C3 + paired bootstrap CIs + Wilcoxon.
- Mechanism: repair-success-rate on the _flagged_ component (C2 vs C1) — the structure-mattered evidence.
- Cost/latency: LLM calls per source, C2 ≪ C3.
- Metric blind-spot: automatic scalar metrics miss the human-visible C2 gain.
- **Honest cut-lines pre-registered** (rq4 protocol §6): H4a/H4b/H4c outcomes each have a framing.

## §7 Deployment / Real-World Eval — **[GATED + scope TBD]**

Source: strategy §3/§7 (narrow high-utility Burmese how-to demo). The eval-data engine. Keep claims to
impact + real (non-synthetic) ratings; not a venture.

## §8 Limitations / Ethics — **[WRITE NOW]**

- Single pair/direction; small expert panel; NC license; repair LLM cost; the generality claim rests
  on one language for now (the "Executable Translation" breadth is positioned as future work unless a
  2nd domain/pair is added).

## §9 Conclusion — **[WRITE LAST]**

---

## Build order (respects the RQ3 gate)

1. **[NOW]** Draft §1 Intro, §2 Related Work, §3 Background, §4 Method prose (all stable). Spin up the
   TACL LaTeX skeleton (reuse `paper/` cite infra + `references.bib`).
2. **[NOW, optional]** Stub the companion WAT paper outline from the WMT-floor material.
3. **[AFTER RQ3 confirms IFS]** Build the C2 localizer→repair agent (smoke-first on a few segments),
   then C1/C3; generate on the 40 sources.
4. **[AFTER the C0–C3 human round]** Fill §5 numbers, §6 Results, §7 Deployment, §9 Conclusion.

**The bottleneck for everything below line 1–2 is the human study (RQ3).** Nothing in §6/§7 can be
honestly written before it runs.
