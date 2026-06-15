# Strategy — Novelty + Deployment for the WikiHow en→my Project

_Living doc. Started 2026-06-07. Author: Pyae (Seon). Status: v1 (pre-research-synthesis)._

This captures the strategic turn: not "first-to-publish fine-tuning," but **novel research that
also ships a practical, high-impact deployment** for Burmese speakers. Three web research agents
were launched 2026-06-07 to ground the open questions; this doc will be updated with their findings.

---

## 1. The question on the table

> "I don't just want first-submit/first-publish work — I want a practical result too (e.g. a
> Burmese WikiHow so Burmese people's search results are full of how-to helpers). And shouldn't it
> be **novel** research rather than just fine-tuning? What about **translation agents** we deploy in
> production for WikiHow Myanmar?"

Short answer: **the instinct is correct.** Plain fine-tuning is a baseline, not a contribution.
The novel, defensible, _and_ deployable direction is an **instruction-faithfulness-guided
translation agent** — which unifies the paper, the product, and the impact.

---

## 2. Honest novelty assessment (no cheerleading)

| Component                                | Novelty     | Verdict                                                                                                     |
| ---------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| Fine-tune NLLB-600M en→my                | ~none       | Baseline. Reviewers have seen it 50×. Necessary, not sufficient.                                            |
| WikiHow en–my MTPE corpus (~10k)         | real        | A genuine resource contribution (low-resource data is valued).                                              |
| IFS metric (step/action/entity/quantity) | modest      | Combination-novel; name taken; precedents exist (M-ETA, WMT critical-error). Solid but not memorable alone. |
| **IFS-guided _agentic_ translation**     | **genuine** | **The differentiator.** Underexplored for low-resource + procedural text.                                   |

The current paper = **resource + benchmark + modest metric** → solid _workshop_ paper, not memorable.
Reaching for the agent is what makes it stand out.

---

## 3. Pushback (requested)

1. **"Email WikiHow to let them use our API" — framing inverted, and likely unnecessary.** We use
   _their_ content under license; they don't need ours. WikiHow content is **CC BY-NC-SA 3.0**, which
   (pending verification) **already permits non-commercial translated derivatives** with attribution
   - sharealike. So a free Burmese how-to site is likely legal **without a partnership** — we just
     can't use the "wikiHow" trademark or monetize. Email them as courtesy/distribution, not as a blocker.

2. **Naive deployment has no moat — Google Translate is the killer competitor.** Burmese users can
   already auto-translate any wikihow.com page for free. "MT + hosting" is a weekend build and Google
   substitutes it. **The only moat is quality**: human-post-edited, _instruction-faithful_,
   Burmese-fluent output that Google Translate provably botches on procedural text (reordered steps,
   dropped quantities, wrong actions). That moat is _exactly what the research produces_ →
   **the research IS the product's moat.**

3. **As a business it fails the weekend-vibe-code test (no buyer).** Nobody pays for free Burmese
   how-to; Myanmar ad market is tiny. **But** as (a) an impact public good, (b) real-user eval data
   that makes the paper credible (not synthetic), and (c) a portfolio/grant piece, it's strong.
   → Pursue as **research-impact + eval-data, NOT as a venture.** Don't over-invest like it's a startup.

---

## 4. The unifying thesis

**Instruction-Faithfulness-Guided Agentic Translation for Low-Resource Procedural Text (English→Burmese).**

- Fine-tuned NLLB = the fast **base translator** inside the agent (not discarded).
- A **translation agent**: translate → reflect/critique → repair. Underexplored for low-resource +
  procedural text (most agentic MT is high-resource or literary).
- **IFS becomes the control signal**, not just a ruler: the agent self-checks preservation of steps,
  actions, entities, quantities, and repairs failures. _IFS-as-reward_ is the novel claim.
- The agent's output = the deployed Burmese how-to corpus. **Paper → product → impact, one pipeline.**

---

## 5. Two-track plan (protect the August deadline)

WMT 2026 deadline ≈ **August 2026 (~2 months)**. Do NOT bolt a full agent onto it and risk the miss.

- **Track 1 — WMT 2026 (now):** corpus + NLLB(zero-shot/fine-tuned)/Gemini/Google-Translate benchmark
  - IFS metric + BURMESE-SAN cross-eval. Tight, shippable. Tease the agent as future work.
- **Track 2 — agent + deployment (next):** IFS-guided agentic translation; live Burmese how-to demo;
  target ACL/EMNLP 2027 + an impact grant (e.g. Lacuna Fund). The novel, higher-impact follow-up.

The fine-tune currently running is **step 1 of Track 2**, not a dead end.

---

## 6. Open questions (being researched 2026-06-07)

- **Novelty gap:** what agentic-MT work already exists; is "IFS-as-control-signal for low-resource
  procedural MT" genuinely unclaimed? (Agent 1)
- **Burmese MT + WikiHow:** SOTA en→my; does wikiHow already have Burmese; how does wikiHow translate;
  any API; exact license. (Agent 2)
- **License + market + funding:** does CC BY-NC-SA 3.0 permit NC translated derivatives; Burmese
  how-to content gap; who funds low-resource language public goods; comparables. (Agent 3)

### Research findings (2026-06-07, 3 web agents — cited)

**Novelty (Agent 1):** Defensible claim = _first MT method to use a structured instruction-faithfulness
score (step ordering + actions + entities + quantities) as an in-loop control/reward signal, for
low-resource PROCEDURAL en→Burmese, where COMET is poorly calibrated._ Differentiate from / cite as
loop ancestors: TEaR (NAACL 2025, MQM-driven), LLMRefine, Andrew Ng translation-agent (free-text
critique), MAPS (TACL 2024, QE selection). Already-done (don't claim): reflection/self-refine,
multi-agent MT, COMET/MetricX as RL reward, glossary/RAG, faithfulness-via-instruction-augmentation
(2308.12674), entity-preservation as _evaluation_ (2203.05227 — must cite, repurposing not inventing).
Procedural translation as a task is a near-void in the literature; "agentic" alone is NOT the novelty —
the reward signal is.

**License (Agents 2+3) — CRITICAL:** wikiHow changed its license on **24 March 2025**; content published
**after that date is proprietary (no longer CC).** Only **pre-March-2025** text is CC BY-NC-SA 3.0.
→ Must verify our corpus was scraped from pre-2025 content (gates dataset legality, not just deployment).
CC BY-NC-SA 3.0 permits translated derivatives IF: non-commercial, ShareAlike (CC BY-NC-SA), attribute
wikiHow, **no trademark/logo/branding**, and verify image terms (re-illustrate or text-only).

**Burmese MT landscape (Agent 2):** wikiHow has **no Burmese edition** (33 langs incl. zh/ja/ko/th/vi/hi).
NLLB en→my FLORES chrF++ ≈25 (our in-domain 36.01 is healthy). Gemini 2.5 Pro leads Burmese MT
(BURMESE-SAN MetricX 90.2); confirms Gemini baseline. chrF > BLEU (unsegmented Burmese, Zawgyi/Unicode).
Other resources: ALT (18k/1k/1k), FLORES+, myXNLI.

**Deployment realities (Agent 3):** Real gap (mobile-first, 24M users). Risks are NOT legal — they are
**sustainability** (Khan Academy volunteer translation stalls → need small PAID post-editing core) and
**access** (Burmese Wikipedia blocked by junta Feb 2021 → host mirrors outside Myanmar). Narrow scope
wins (TWB Rohingya glossary). Funding: Wikimedia grants ($5k rapid / $10k-550k) best fit; Lacuna/
Masakhane currently Africa-skewed; CLEAR Global/TWB already work in Burmese (partner, not funder).

---

## 7. Plan (revised post-research)

**Two-paper arc (not "settling" — staking priority then swinging big):**

- **WMT 2026 (Aug, nearly free):** corpus + NLLB/Gemini/GTranslate benchmark + **IFS-as-metric** +
  BURMESE-SAN cross-eval. Stakes the dataset + IFS claim publicly. Only realistic 2026 venue.
- **ACL 2027 flagship (~Feb deadline):** **IFS-as-control-signal** agentic translation for low-resource
  procedural text + the deployment as real-world validation. The memorable paper.

**Deployment = narrow high-utility demo** (not a full site): produces the IFS human-correlation eval the
flagship needs + a demo + a grant seed (Wikimedia/TWB). One effort, three payoffs. Paid post-editing
core, Unicode (+Zawgyi convert), mirrors outside Myanmar.

**WikiHow:** proceed under CC BY-NC-SA (pre-2025 content only, NC, SA, attribute, no trademark); email
later as courtesy. (User decision 2026-06-07.)

## 8. Decisions locked (2026-06-07)

- **Venue = two-paper arc.** WMT 2026 (Aug) floor: corpus + NLLB/Gemini/GTranslate benchmark +
  IFS-as-metric + BURMESE-SAN cross-eval. ACL 2027 flagship: IFS-as-control-signal agentic translation
  for low-resource procedural text + deployment validation.
- **Deployment = narrow high-utility demo** (one how-to category). Serves the flagship's IFS
  human-correlation study + a demo + a Wikimedia/TWB grant seed.
- **WikiHow = proceed under CC BY-NC-SA**, email later as courtesy. No trademark/branding.
- **Corpus = scraped BEFORE 24 March 2025 → CC BY-NC-SA, legally clear.** (Confirmed by user.)

## 9. Open / next

- Confirm WMT 2026 exact deadline (statmt.org/wmt26).
- Finish the running fine-tune (Track-1 headline result).
- Track-1 remaining: Gemini 2.5 + Google Translate baselines; IFS metric + human-validation; BURMESE-SAN
  / FLORES+ cross-eval; write-up.
- Track-2 flagship design: IFS-as-reward agent loop (base = fine-tuned NLLB; LLM reflect/repair; IFS
  gates refinement + termination). Differentiate from TEaR / LLMRefine / Ng / MAPS.

## 10. Flagship sharpened — v2 (2026-06-12, post-WMT-negative-result)

**The locked v1 thesis ("IFS-as-reward") is half-dead and we should say so.** Our own WMT reranking
experiment proved structural IFS is **near-flat across the fine-tuned model's 16-best** (paraphrases
that preserve the same steps/entities/quantities); COMET-Kiwi's per-source ranking correlates only
**ρ=0.10** with reference chrF++; scalar reranking recovered **+0.41** of a **+5.3** oracle headroom.
A reviewer who read our WMT paper would kill "IFS-as-scalar-reward" in one line.

**The pivot:** IFS is a poor _ranker_ of similar candidates but an excellent _structured error
LOCALIZER_ (which step dropped, which quantity changed, which entity vanished). So the flagship is
**IFS-as-structured-error-localizer → selective, constraint-targeted repair**, and the WMT negative
result becomes the flagship's _motivating experiment_: the field's standard move (scalar quality
signal — COMET-reward RL, QE reranking, MBR) _provably_ cannot recover the headroom for low-resource
Burmese; that failure is _why_ structured localizing repair is needed.

**Thesis spine (chosen 2026-06-12, layered):** PhD thesis + SOP narrative = **"Executable
Translation"** (procedural/executable MT as a problem class — followability, not BLEU; generalizes to
medical/legal/safety MT); the single ACL/EMNLP 2027 paper = **"When Metrics Fail"** (when the quality
metric is unreliable — the default for low-resource — stop optimizing a scalar reward; decompose
faithfulness into checkable structured constraints, localize the break, repair it selectively).

**Why structured beats free-text critique for low-resource specifically (the intellectual core):** a
free-text LLM critic is _itself_ miscalibrated on Burmese fluency (same low-resource failure that
breaks COMET); structured constraints (this numeral, this entity, this imperative, this step) are
**language-agnostic, checkable surface anchors** — verifying them is cross-lingual _matching_, not
Burmese _understanding_.

**Selective repair = method novelty AND deployment necessity:** NLLB runs on-device offline in
low-connectivity Myanmar; the LLM fires _only_ on IFS-flagged segments (~k%) → cheap, degrades
gracefully. Answers the "why not just use the LLM?" reviewer directly.

**The decisive experiment + cut-lines:** C0 base / C1 free-text-critique repair / C2 IFS-localized
repair / C3 full-LLM, scored on **human followability**. Holds iff **C2 > C1 ≥ C0 at a fraction of
C3's cost**. If C1≈C2 → structure didn't matter (cost-efficiency paper); if C3 dominates cheaply →
"just use the LLM" finding (pivot to on-device angle). Full design: `docs/flagship-rq4-protocol.md`.

**Gating dependency:** flagship is gated on RQ3 (IFS validating against human followability in the
WMT study). The human study is the flagship's _foundation_, not just WMT's keystone. **Not built yet
(user decision 2026-06-12): documented + instrument made forward-compatible; do not implement the
agent until RQ3 confirms.** moat-check intentionally skipped — this is a research thesis, not a
venture (deployment = impact + eval-data by design; see §3).

## 11. Venue decision — flagship + companion (2026-06-12, research-backed)

Two web-research agents surveyed ACL/ARR 2027 timelines + where the sibling papers actually land.
Key facts: it's an **ARR world** (ACL/EMNLP/NAACL/EACL all commit from ACL Rolling Review, so the
letterhead is largely interchangeable — cycle + audience matter more than name). Closest published
precedents are **NOT at ACL**: MAPS/xCOMET/TransAgents → **TACL**; AfriCOMET (low-resource metric +
human validation) + TEaR (translate→estimate→refine) → **NAACL main**.

- **Flagship target = TACL (primary), NAACL 2027 main (fallback).** (User chose, over ACL/EMNLP.)
  Rationale: TACL is **rolling** (submit the month the human study + C2>C1 result lands — matches the
  flagship's RQ3-gated, uncertain ready-date; no deadline cliff), is the **exact home of the sibling
  agentic-MT + metric papers**, gives unbounded space for the full human protocol, and is = ACL/EMNLP
  main prestige to PhD committees. NAACL 2027 main (ARR **~Oct 2026** cycle → commit ~Jan 2027;
  Pittsburgh, May 2027) is the precedent-perfect conference fallback if a stage/faster visibility is
  wanted — but that deadline is TIGHT given RQ3 isn't run. ACL 2027 (the old plan) is the
  latest-timing option (ARR ~Feb 2027, Japan Jul 2027) and **no better a fit** — don't anchor on the
  famous name. **Do NOT put the flagship at WMT** (underselling; WMT = the floor paper).
- **Companion = WikiHow-MY corpus → WAT @ EMNLP** (Workshop on Asian Translation; Burmese–English is a
  recurring WAT pair). 4–8pp resource paper carved from the WMT-floor material (mostly written) →
  Burmese/Asian-MT community visibility + a citeable resource that strengthens the flagship's claims.
- Avoid for the core paper: WMT Metrics shared task (system-descriptions only), LREC-COLING
  (too low-prestige given AfriCOMET landed NAACL main), workshop-only for the methodology.
- ARR mechanics: ~10-week cycles, 2026 cycles Mar 16 / May 25 / Aug 3 / Oct 12; submit→reviews
  (~10wk)→commit to a venue→PC decision (~4–6wk). EMNLP 2026 (ARR May 25) already past.
