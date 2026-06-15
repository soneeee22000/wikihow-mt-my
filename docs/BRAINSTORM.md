# Brainstorm Session: Flagship (ACL/EMNLP 2027) — PhD-caliber follow-up

Date: 2026-06-12
Facilitator pass: adversarial pressure-test (requested), not coached-from-scratch.

## Problem Space

The WMT 2026 paper (corpus + benchmark + IFS-as-metric + reranking negative result) is the
**floor** — a solid resource paper, not a thesis-defining one. We need the flagship: a genuinely
novel research technique, deployable, that anchors a top-5 PhD narrative. Working seed (RQ4): use
IFS as a control signal in an agentic translate→reflect→repair loop for low-resource procedural MT
(English→Burmese).

## Technique Used

**Reverse / adversarial pressure-test on the seed.** We did not generate options blind — we took
the locked seed and attacked it with our own just-produced evidence, then rebuilt the sharper form
that survives. (First Principles on "what is IFS actually good for" did the rebuild.)

## The pivot (the session's core insight)

Our WMT reranking experiment **invalidates the naive flagship** and **hands us a stronger one**:

- Measured: structural IFS is **near-flat across the fine-tuned model's 16-best** (paraphrases that
  preserve the same steps/entities/quantities); COMET-Kiwi's per-source ranking correlates only
  **ρ=0.10** with reference chrF++; scalar reranking recovered **+0.41** of a **+5.3** oracle headroom.
- Therefore **"IFS-as-scalar-reward" is dead on arrival** — IFS cannot separate similar candidates.
- But IFS is an excellent **structured error LOCALIZER**: it names _which_ step dropped, _which_
  quantity changed, _which_ entity vanished.
- **Flagship pivots: IFS-as-reward → IFS-as-structured-error-localizer driving targeted repair.**
  The negative result stops being a wart and becomes the **motivating experiment**: the field's
  standard move (scalar quality signal — COMET-reward RL, QE reranking, MBR) _provably_ cannot
  recover the headroom for low-resource Burmese; that failure is _why_ structured localizing repair
  is needed.

## Thesis spine (chosen: layered)

**Executable Translation** (PhD thesis + SOP narrative) → **"When Metrics Fail"** (the single ACL
2027 paper). When the quality metric is unreliable — the _default_ for low-resource languages, not
the exception — stop optimizing a scalar reward; decompose faithfulness into checkable structured
constraints, **localize** the break, and **repair** it selectively.

Three legs, each half-built:

1. **Problem** — procedural/instructional MT where the failure mode is _unfollowability_ (dropped
   negation, wrong dose, reordered step), not BLEU. Under-studied; high-stakes; worst for low-resource.
2. **Diagnosis (WMT paper)** — learned metrics miscalibrated for low-resource Burmese, shown three
   ways: benchmark disagreement, FLORES+ control, reranking negative result.
3. **Method (flagship)** — IFS as structured localizer → selective, constraint-targeted repair in an
   agent; validated against **real human followability** from the deployment.

## Ideas Generated

### Theme 1: Framing

- **Executable Translation (thesis) + "When Metrics Fail" (paper)** — CHOSEN. Memorable thesis,
  defensible single paper.
- "When Metrics Fail" only — tight, lower ceiling.
- "Executable Translation" broad — highest ceiling, must defend generality (>1 domain/pair).
- "Deployment-as-Research" — impact/grant strength, weaker as pure-method ML.

### Theme 2: Method mechanics

- IFS-as-localizer emits a **structured repair instruction** ("step 3 dropped 350°F; regenerate
  preserving it"), not a scalar.
- **Selective repair**: NLLB on-device offline; LLM fires only on IFS-flagged segments (~k%) →
  cheap, degrades gracefully without connectivity. Method novelty _and_ deployment necessity.
- Why structured beats free-text critique **for low-resource specifically**: a free-text LLM critic
  is _also_ miscalibrated on Burmese fluency, but structured constraints (this numeral, this entity,
  this step) are **language-agnostic, checkable surface anchors** that don't require the critic to
  "understand" Burmese. ← the intellectual core.

### Theme 3: Validation / eval engine

- One human-followability study, **two papers**: instrument the WMT study so the flagship's repair
  conditions are rated by the _same calibrated raters_ on the _same 40 sources_ (zero re-baselining).
- Add **binary component checks** (step/action/entity/quantity Y/N) to the rating UI — WMT needs the
  human "action" component anyway; flagship needs "did the repair fix the flagged element?".

### Theme 4: Risks surfaced (the honest cut-lines)

- **A — "why not translate with the LLM directly?"** Answer = selective-repair cost/offline; must
  _measure_ structured-selective ≥ full-LLM at a fraction of cost.
- **B — IFS-localized repair vs free-text critique** (the whole ballgame). Decided by the C0–C3 study.
- **D — IFS could fail WMT human validation (RQ3)** → flagship's foundation, not just WMT's keystone.

## Top 3 Ranked Ideas

1. **IFS-as-structured-localizer → selective constraint-targeted repair** — the method. Negative
   result motivates it; principled reason it beats free-text for low-resource. | Feasibility: **Med**
   (gated on RQ3 + the C0–C3 experiment landing C2>C1).
2. **One human study, two papers (forward-compatible instrument)** — cheap, high-leverage, do now. |
   Feasibility: **High**.
3. **"Executable Translation" thesis framing** — the narrative that makes it memorable to a committee
   - generalizes to medical/legal/safety MT. | Feasibility: **High** (framing, not engineering).

## Recommended Next Step

Standard `/brainstorm` routes the top idea through `/moat-check`. **Skipped deliberately, with
reason:** this is a _research thesis_, not a product venture. Per `strategy-novelty-and-deployment.md`
§3, the deployment is explicitly research-impact + eval-data, **not** a business (no buyer; fails the
weekend-vibe test _by design_, and that's intentional). Gating a PhD research direction on
product-market fit is a category error. The novelty is defended via the cited literature review
(white-space confirmed), not a moat-check. The _only_ go/no-go gate that matters here is empirical:
**RQ3 (does IFS validate against human followability?) → then RQ4 (does C2 > C1 ≥ C0?).**

Action taken this session (see `docs/flagship-rq4-protocol.md`): designed the RQ4 experiment and
made the WMT human-rating instrument forward-compatible so it collects the flagship's data.
