# WMT 2026 — Submission Readiness Checklist

**Date:** 2026-06-17. Cross-checked against the WMT 2026 call (statmt.org/wmt26) + ACL/ARR norms.

## The two things you flagged

1. **"Numbers 1–569 beside every sentence" — NOT a bug.** That is the ACL/WMT
   `\usepackage[review]{acl}` margin line-numbering. It is **mandatory** for the
   anonymised review version so reviewers can cite "line 234". It disappears automatically
   at camera-ready (`[review]` → `[final]`). I did not leave an error in.
   - **Clean reading copy with no line numbers:** `paper/main_reading.pdf` (built from
     `paper/main_reading.tex`, `[final]` mode, author still anonymous). Read that one.
   - **Submit `paper/main.pdf`** (the line-numbered one) to WMT.

2. **GitHub / HuggingFace / Kaggle links — present but deliberately withheld.** WMT 2026's
   **research track is anonymised** ("papers should be anonymised", 6–10 pages). Any real
   author name or resource link in the _submitted_ PDF = de-anonymisation = desk-reject risk.
   So all four links are staged in the camera-ready comment block in `main.tex` and excluded
   from the review PDF — which is correct:
   - Model: `https://huggingface.co/PyaeSoneK/nllb-600m-wikihow-en-my` ✓
   - Collection: `https://huggingface.co/collections/PyaeSoneK/english-to-burmese-mt-wikihow-...` ✓
   - Code: `https://github.com/soneeee22000/wikihow-mt-my` ✓
   - Kaggle (now filled): train `…/wikihow-my-ft`, eval `…/wikihow-metricx-run`,
     rerank `…/wikihow-nbest-run` (all under `kaggle.com/code/pyaesonekyaw/`)

## Fixed this session

- ✅ Kaggle placeholder resolved → 3 real notebook URLs staged for camera-ready.
- ✅ **Limitations** moved to `\section*{Limitations}` (unnumbered) **after** Conclusion,
  before References — ACL convention (was numbered, before Conclusion).
- ✅ Added `\section*{Ethics Statement}` (annotator consent, no PII, aggregate-only release,
  CC BY-NC-SA rehydration, deployment caution) — expected for a human-study + NC-source paper.
- ✅ Rebuilt: **8 pages, 0 undefined refs/cites, 0 overfull hboxes**, both variants anonymous.

## Cross-check findings (WMT 2026 vs. our paper)

| Item                                                                 | Status                                                                                                                            |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Anonymisation (research track)                                       | ✅ correct — links/author withheld                                                                                                |
| Page limit                                                           | ✅ **6–10 content pages** (not 8) — we're ~7, lots of headroom                                                                    |
| Limitations section                                                  | ✅ added/repositioned (unnumbered)                                                                                                |
| Ethics statement                                                     | ✅ added                                                                                                                          |
| Metrics: COMET + MetricX-24, bootstrap CIs, Williams, Krippendorff α | ✅ all current WMT best-practice                                                                                                  |
| chrF++ as _primary_ headline metric                                  | ⚠️ mildly off-trend — defensible for unsegmented Burmese (we justify it), but consider leading headline claims with COMET/MetricX |

## Remaining — YOUR actions (I can't/shouldn't do these)

1. **Make the 3 Kaggle notebooks Public** before camera-ready (one click each in Kaggle →
   Settings → Sharing). Private = 403 for readers. _Not needed for the review submission._
2. **(Optional, review version):** if you want reviewers to see code, add an
   `anonymous.4open.science` mirror of the repo as an anonymised footnote. Otherwise the
   current "withheld for review" stance is fine.
3. **(Optional polish):** a short explicit Data Statement (Bender & Friedman) — annotator
   variety/guidelines. Most of it is already in §3 (Corpus) + the new Ethics statement.
4. **Submit `paper/main.pdf` when the WMT 2026 research-track portal opens.** Re-confirm the
   exact page sentence on statmt.org/wmt26 at submission time (calls get updated).

## Camera-ready flip (when accepted)

In `main.tex`: `[review]` → `[final]`, uncomment the real `\author{…}` block, and move the
four resource links into a footnote/Availability paragraph. Make Kaggle notebooks public.
