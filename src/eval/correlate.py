"""Validate IFS against human instruction-followability ratings (Track-1 P3 gate).

Two modes:
  --make-sheet : sample segments across systems' hypotheses into a blank, blind,
                 shuffled ratings CSV for human raters (one row per system x segment).
  (analysis)   : read a filled ratings CSV, compute per-segment automatic metrics
                 (IFS, sentence chrF++, sentence BLEU; COMET read from a `comet`
                 column if present) and report Pearson/Spearman of each metric vs the
                 human followability rating, plus a Williams test comparing IFS
                 against each competing metric on the shared human variable.

Usage:
  python src/eval/correlate.py --make-sheet \
      --systems nllb_zeroshot,nllb_finetuned,gemini,gtranslate --segments 40
  python src/eval/correlate.py --ratings experiments/results/ratings_filled.csv
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import sacrebleu
from scipy import stats

import ifs  # same package dir (src/eval is sys.path[0] when run as a script)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "experiments", "results")
SEED = 42
LIKERT_MIN, LIKERT_MAX = 1, 5
DEFAULT_SHEET = os.path.join(RESULTS, "ratings_sheet.csv")
DEFAULT_KEY = os.path.join(RESULTS, "ratings_key.csv")
# Blind sheet the rater fills (no system, no reference, opaque id):
SHEET_FIELDS = ["id", "src_en", "hyp_my", "followability", "adequacy"]
# Private key (system + reference), joined by id at analysis time only:
KEY_FIELDS = ["id", "system", "ref_my"]


def read_rows(split: str) -> list:
    """Load split rows ({en, my, ...}) from data/processed/<split>.jsonl."""
    with open(os.path.join(PROC, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_hyps(system: str, split: str) -> list:
    """Load one-per-line hypotheses for a system/split from experiments/results."""
    path = os.path.join(RESULTS, f"{system}_{split}_hyps.txt")
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def make_sheet(systems: list, split: str, n_segments: int, out: str, key_out: str) -> None:
    """Write a TRULY BLIND, shuffled ratings sheet (opaque ids; system + reference
    hidden in a separate key file) so ratings can't be biased by which system or by
    the reference. Same source segments across systems for paired comparison."""
    rows = read_rows(split)
    idx = random.Random(SEED).sample(range(len(rows)), min(n_segments, len(rows)))
    records = []
    for system in systems:
        hyps = read_hyps(system, split)
        for i in idx:
            records.append({"system": system, "src_en": rows[i]["en"],
                            "ref_my": rows[i]["my"], "hyp_my": hyps[i]})
    random.Random(SEED + 1).shuffle(records)  # blind raters to system order

    sheet, key = [], []
    for n, rec in enumerate(records, 1):
        rid = f"seg-{n:04d}"
        sheet.append({"id": rid, "src_en": rec["src_en"], "hyp_my": rec["hyp_my"],
                      "followability": "", "adequacy": ""})
        key.append({"id": rid, "system": rec["system"], "ref_my": rec["ref_my"]})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for path, fields, data in ((out, SHEET_FIELDS, sheet), (key_out, KEY_FIELDS, key)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    print(f"wrote {len(sheet)} rows -> {out} (fill followability + adequacy, "
          f"{LIKERT_MIN}-{LIKERT_MAX}); system/reference key -> {key_out}")


def sentence_scores(src: str, ref: str, hyp: str) -> dict:
    """Per-segment automatic metrics: IFS (source-anchored), chrF++ and BLEU (vs ref)."""
    return {
        "ifs": ifs.segment_ifs(src, hyp)["ifs"] * 100,
        "chrf": sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score,
        "bleu": sacrebleu.sentence_bleu(hyp, [ref]).score,
    }


def williams_test(r_a: float, r_b: float, r_ab: float, n: int) -> dict:
    """Two-sided Williams test for dependent correlations sharing the human variable.

    r_a, r_b = correlations of metric A and metric B with the human rating;
    r_ab = correlation between metric A and metric B. Tests r_a != r_b.
    """
    det = 1 - r_a**2 - r_b**2 - r_ab**2 + 2 * r_a * r_b * r_ab
    df = n - 3
    den = 2 * ((n - 1) / df) * det + ((r_a + r_b) ** 2 / 4) * (1 - r_ab) ** 3
    t = (r_a - r_b) * ((n - 1) * (1 + r_ab) / den) ** 0.5
    return {"t": round(t, 4), "p": round(2 * stats.t.sf(abs(t), df), 5), "df": df}


def correlate(metric_vals: list, human: list) -> dict:
    """Pearson and Spearman of a metric against the human ratings, with p-values."""
    pr = stats.pearsonr(metric_vals, human)
    sr = stats.spearmanr(metric_vals, human)
    return {"pearson": round(float(pr.statistic), 4), "pearson_p": round(float(pr.pvalue), 5),
            "spearman": round(float(sr.statistic), 4), "spearman_p": round(float(sr.pvalue), 5)}


def analyze(ratings_path: str, key_path: str) -> dict:
    """Compute metric-vs-human correlations + Williams tests from a filled blind sheet,
    joining the private key (id -> system, ref_my) for the reference and system labels."""
    with open(key_path, encoding="utf-8") as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    with open(ratings_path, encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            if not r.get("followability", "").strip():
                continue
            k = key.get(r["id"], {})
            rows.append({**r, "ref_my": k.get("ref_my", ""), "system": k.get("system", "")})
    if not rows:
        raise ValueError("no rated rows (fill the 'followability' column 1-5)")
    human = [float(r["followability"]) for r in rows]
    metrics = {"ifs": [], "chrf": [], "bleu": []}
    has_comet = all(r.get("comet", "").strip() for r in rows)
    if has_comet:
        metrics["comet"] = [float(r["comet"]) for r in rows]
    for row in rows:
        scored = sentence_scores(row["src_en"], row["ref_my"], row["hyp_my"])
        for name in ("ifs", "chrf", "bleu"):
            metrics[name].append(scored[name])

    corr = {name: correlate(vals, human) for name, vals in metrics.items()}
    williams = {}
    for name, vals in metrics.items():
        if name == "ifs":
            continue
        r_ab = float(np.corrcoef(metrics["ifs"], vals)[0, 1])
        test = williams_test(corr["ifs"]["pearson"], corr[name]["pearson"], r_ab, len(human))
        test["ifs_higher"] = corr["ifs"]["pearson"] > corr[name]["pearson"]
        williams[f"ifs_vs_{name}"] = test
    return {"n": len(human), "correlations": corr, "williams_ifs_vs": williams,
            "comet_included": has_comet}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sheet", action="store_true")
    ap.add_argument("--systems", default="nllb_zeroshot,nllb_finetuned,gemini,gtranslate")
    ap.add_argument("--segments", type=int, default=40, help="source sentences to sample")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sheet-out", default=DEFAULT_SHEET)
    ap.add_argument("--key-out", default=DEFAULT_KEY)
    ap.add_argument("--ratings", help="filled ratings CSV (analysis mode)")
    ap.add_argument("--key", default=DEFAULT_KEY, help="private id->system/ref key")
    ap.add_argument("--out", default=os.path.join(RESULTS, "ifs_correlation.json"))
    args = ap.parse_args()

    if args.make_sheet:
        make_sheet(args.systems.split(","), args.split, args.segments, args.sheet_out, args.key_out)
        return
    if not args.ratings:
        ap.error("pass --ratings <filled.csv> for analysis, or --make-sheet to build one")
    result = analyze(args.ratings, args.key)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
