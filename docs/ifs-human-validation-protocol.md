# IFS Human-Validation Protocol (Track-1 Phase 3 gate)

_Created 2026-06-08. Validates the paper's novel claim: that automatic **IFS** tracks human judgment
of **instruction-followability** at least as well as — and on the instruction-specific sub-judgment,
better than — chrF/COMET/BLEU. Tooling: `src/eval/correlate.py`. Detail context: `docs/spec.md`._

## 1. Claim under test

> Automatic IFS (quantity + entity + step; action via this human study) correlates with human
> instruction-followability ratings **significantly better than COMET/chrF on the step+action
> sub-judgment**, and competitively overall — making it a useful procedural-MT metric and a control
> signal for the Track-2 agent.

Honesty: if COMET wins **overall**, report it. IFS can still win on the **instruction-specific**
sub-judgment (can a reader actually perform the step correctly?), which is the contribution.

## 2. What humans rate (1–5 Likert, anchored)

Per segment the rater sees the **English source** + a **Burmese system output** (reference hidden,
system label hidden, order randomised), and rates:

- **Adequacy** — overall meaning preserved. 1 = meaning lost; 5 = fully preserved.
- **Instruction-Followability** (primary) — _could a Burmese reader carry out this step correctly and
  in the right order from the translation alone?_ 1 = would fail / do the wrong action; 3 = mostly
  right, minor ambiguity; 5 = perfectly followable.
- **Component checks (binary, now MANDATORY in the UI)** — to validate each IFS component against a
  human counterpart: step-order preserved (Y/N), action/verb correct (Y/N), entities correct (Y/N),
  quantities/units correct (Y/N). The **action** check is the only human-only IFS component. Promoted
  from "optional" to mandatory 2026-06-12 because (a) WMT needs the human action component, and (b) the
  flagship measures whether a repair fixed the _specific flagged_ component — one instrument, two
  papers (see `docs/flagship-rq4-protocol.md`). Collected via `<Choices>` widgets in
  `ratings_ls_config.xml`; exported by `scripts/ls_export_to_csv.py`.

## 3. Sampling

- **K source sentences** sampled (seed 42) from `test`; for each, include **every system's** output
  → `K × |systems|` rated rows (paired across systems → enables paired comparison). Default K = 40
  (≈160 rows for 4 systems — a realistic author load).
- Stratify lightly by source length if needed; otherwise uniform random.
- Built blind + shuffled by `python src/eval/correlate.py --make-sheet`.

## 4. Raters & reliability

- **Author** rates the full sheet (native speaker).
- **2–3 light raters** (SpeakProof network / Ekkhara / family) rate a **~50-row overlap subset**
  → one **Krippendorff's α** (ordinal). This single number converts "author opinion" into a
  validated instrument and pre-empts the solo-rater critique (the #1 IFS reviewer risk).
- **Author test–retest**: re-rate 50 rows ≥2 weeks apart → intra-rater reliability.
- Fallback if recruiting fails: solo + test–retest, named as the headline limitation.

## 5. Procedure

- Raters get a one-page instruction sheet + 3 worked anchor examples (1/3/5).
- Judge the **translation as an instruction**, not its literary polish. Penalise: wrong/omitted
  action, reordered steps, dropped/altered numbers, wrong entity. Ignore: dialectal word choice that
  a reader would still follow correctly.
- Fill `followability` and `adequacy` columns (and component Y/N if used); leave other columns intact.

## 6. Analysis (`correlate.py`, default mode)

For each metric (IFS, chrF++, BLEU, and COMET if a per-segment `comet` column is supplied) vs the
human **followability**:

- **Pearson r** and **Spearman ρ** (segment-level), with p-values.
- **Williams (Hotelling–Williams) test** comparing IFS's correlation against each competing metric's
  (dependent correlations sharing the human variable) → is IFS significantly better/worse?
- Report system-level means too (each system's mean human followability vs mean metric).

## 7. Deliverables & gate

- `experiments/results/ratings_sheet.csv` (blank) → filled `ratings_filled.csv`.
- Krippendorff α (or test–retest) number.
- `correlate.py` correlation + Williams table (JSON + LaTeX-ready).
- **Gate (P3):** IFS shows significant positive correlation with human followability and is
  **≥ COMET on the step+action sub-judgment** (Williams n.s.-or-better) — or an honest mixed result
  documented. If IFS doesn't beat BLEU on correlation by end of the IFS window, fall back to the
  dataset+benchmark resource paper (contrib 1–2).
