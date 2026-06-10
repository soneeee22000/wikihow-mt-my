"""Reference-bias signature test for the WikiHow en->my MT corpus.

Tests whether human-post-edited references were derived from Google Translate
output by checking, per system, how closely each system reproduces the
reference (exact match + char-level chrF near-copy rates). A genuinely
independent system almost never reproduces the exact reference; a system used
as the post-edit base would show an anomalously high exact-match / chrF>=90
rate. Read-only over data and result files.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "data" / "processed" / "test.jsonl"
HYP_DIR = ROOT / "experiments" / "results"
SYSTEMS = {
    "nllb_zeroshot": "nllb_zeroshot_test_hyps.txt",
    "nllb_finetuned": "nllb_finetuned_test_hyps.txt",
    "gemini": "gemini_test_hyps.txt",
    "gtranslate": "gtranslate_test_hyps.txt",
}
NEAR_COPY_THRESHOLDS = (90.0, 80.0, 70.0)

# chrF (char-level, no word n-grams) and chrF++ (adds word bigrams).
CHRF_PLAIN = CHRF(char_order=6, word_order=0, beta=2)
CHRF_PP = CHRF(char_order=6, word_order=2, beta=2)


def read_refs(path: Path) -> list[str]:
    refs: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            refs.append(json.loads(line)["my"])
    return refs


def read_hyps(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def seg_chrf(metric: CHRF, hyp: str, ref: str) -> float:
    return metric.sentence_score(hyp, [ref]).score


def rate(values: list[float], threshold: float) -> float:
    return 100.0 * sum(1 for v in values if v >= threshold) / len(values)


def analyse(name: str, hyps: list[str], targets: list[str]) -> dict:
    assert len(hyps) == len(targets), f"{name}: {len(hyps)} != {len(targets)}"
    exact = 0
    pp_scores: list[float] = []
    plain_scores: list[float] = []
    for hyp, tgt in zip(hyps, targets):
        h, t = hyp.strip(), tgt.strip()
        if h == t:
            exact += 1
        pp_scores.append(seg_chrf(CHRF_PP, h, t))
        plain_scores.append(seg_chrf(CHRF_PLAIN, h, t))
    return {
        "name": name,
        "n": len(hyps),
        "exact_pct": 100.0 * exact / len(hyps),
        "exact_count": exact,
        "pp_ge": {th: rate(pp_scores, th) for th in NEAR_COPY_THRESHOLDS},
        "plain_ge": {th: rate(plain_scores, th) for th in NEAR_COPY_THRESHOLDS},
        "pp_mean": statistics.mean(pp_scores),
        "pp_median": statistics.median(pp_scores),
        "plain_mean": statistics.mean(plain_scores),
        "plain_median": statistics.median(plain_scores),
    }


def print_table(title: str, rows: list[dict], key_ge: str, key_mean: str,
                key_median: str) -> None:
    print(f"\n=== {title} ===")
    header = (f"{'system':<16}{'exact%':>8}{'(n)':>7}"
              f"{'>=90%':>8}{'>=80%':>8}{'>=70%':>8}{'mean':>8}{'median':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<16}{r['exact_pct']:>8.2f}{r['exact_count']:>7}"
              f"{r[key_ge][90.0]:>8.2f}{r[key_ge][80.0]:>8.2f}"
              f"{r[key_ge][70.0]:>8.2f}{r[key_mean]:>8.2f}{r[key_median]:>8.2f}")


def main() -> None:
    refs = read_refs(REF_PATH)
    hyps = {name: read_hyps(HYP_DIR / fn) for name, fn in SYSTEMS.items()}
    print(f"n refs = {len(refs)}")
    for name, h in hyps.items():
        print(f"  {name}: {len(h)} lines")

    # Diagnostic 1-4: each system vs the Myanmar reference.
    vs_ref = [analyse(name, hyps[name], refs) for name in SYSTEMS]
    print_table("Systems vs REFERENCE  (chrF++ / word_order=2)",
                vs_ref, "pp_ge", "pp_mean", "pp_median")
    print_table("Systems vs REFERENCE  (plain chrF / word_order=0)",
                vs_ref, "plain_ge", "plain_mean", "plain_median")

    # Diagnostic 5: each system vs Google Translate output.
    gt = hyps["gtranslate"]
    vs_gt = [analyse(name, hyps[name], gt)
             for name in SYSTEMS if name != "gtranslate"]
    print_table("Systems vs GTRANSLATE output  (chrF++ / word_order=2)",
                vs_gt, "pp_ge", "pp_mean", "pp_median")

    # Reference vs Google Translate output (is the ref GT-like?).
    ref_vs_gt = analyse("ref_vs_gt", refs, gt)
    print("\n=== REFERENCE vs GTRANSLATE output ===")
    print(f"exact-match: {ref_vs_gt['exact_pct']:.2f}% "
          f"({ref_vs_gt['exact_count']}/{ref_vs_gt['n']})")
    print(f"chrF++ >=90/80/70: {ref_vs_gt['pp_ge'][90.0]:.2f} / "
          f"{ref_vs_gt['pp_ge'][80.0]:.2f} / {ref_vs_gt['pp_ge'][70.0]:.2f}")
    print(f"chrF++ mean/median: {ref_vs_gt['pp_mean']:.2f} / "
          f"{ref_vs_gt['pp_median']:.2f}")


if __name__ == "__main__":
    main()
