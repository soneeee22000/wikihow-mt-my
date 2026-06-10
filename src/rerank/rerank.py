"""IFS+QE fused reranking of the fine-tuned NLLB N-best list.

Reads the N-best candidates and their reference-free QE scores, computes a
structural IFS score per candidate (reused from src/eval/ifs.py), normalizes QE
and IFS per source, fuses them, and selects one hypothesis per source. Emits one
hypotheses file per variant so the existing automatic.py / COMET / MetricX harness
scores them unchanged:

  rerank_beam1   top beam (baseline; tracks nllb_finetuned)
  rerank_qe      argmax QE only (alpha=0)
  rerank_ifs     argmax IFS only (alpha=1)   [never evaluate with IFS: circular]
  rerank_fused   argmax (1-alpha)*QE + alpha*IFS   (alpha tuned on dev)
  rerank_oracle  argmax reference chrF++ (headroom upper bound; not a system)

Usage:
  python src/rerank/rerank.py --tune --split dev          # grid-search alpha on dev
  python src/rerank/rerank.py --split test --alpha 0.3    # write test variant hyps
  python src/rerank/rerank.py --split test --alpha 0.3 --limit 8   # smoke
"""
import argparse
import json
import os
import sys

import sacrebleu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
from ifs import segment_ifs  # noqa: E402  (sibling module, not a package)

PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "experiments", "results")
NBEST = os.path.join(RESULTS, "nbest")
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
_CHRF = sacrebleu.CHRF(word_order=2)  # chrF++


def read_jsonl(path: str) -> list:
    """Load a list of dict rows from a jsonl file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_split(split: str, limit: int) -> tuple:
    """Return (sources, references) from data/processed/<split>.jsonl."""
    rows = read_jsonl(os.path.join(PROC, f"{split}.jsonl"))
    if limit:
        rows = rows[:limit]
    return [r["en"] for r in rows], [r["my"] for r in rows]


def normalize(values: list) -> list:
    """Per-source min-max to [0,1]; all-equal -> all 0.0 (keeps argmax stable)."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def argmax(scores: list) -> int:
    """Index of the max; ties resolved to the lowest index (best beam first)."""
    best_i, best_v = 0, scores[0]
    for i, v in enumerate(scores):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def candidate_signals(src: str, candidates: list, qe: list) -> dict:
    """Per-candidate raw IFS and QE, plus their per-source normalized forms."""
    ifs = [segment_ifs(src, c)["ifs"] for c in candidates]
    return {"ifs": ifs, "qe": qe,
            "ifs_norm": normalize(ifs), "qe_norm": normalize(qe)}


def pick(variant: str, sig: dict, alpha: float) -> int:
    """Selected candidate index for a variant given its normalized signals."""
    if variant == "beam1":
        return 0
    if variant == "qe":
        return argmax(sig["qe_norm"])
    if variant == "ifs":
        return argmax(sig["ifs_norm"])
    if variant == "fused":
        fused = [(1 - alpha) * q + alpha * f
                 for q, f in zip(sig["qe_norm"], sig["ifs_norm"])]
        return argmax(fused)
    raise ValueError(variant)


def oracle_pick(candidates: list, ref: str) -> int:
    """Candidate index maximizing sentence chrF++ against the reference."""
    scores = [_CHRF.sentence_score(c, [ref]).score for c in candidates]
    return argmax(scores)


def mbr_pick(candidates: list) -> int:
    """MBR: candidate with highest mean chrF++ to the other candidates (pseudo-refs).

    Reference-free consensus decoding (Eikema & Aziz; Freitag 2022) — the standard
    way to cash in N-best headroom without a learned QE model."""
    n = len(candidates)
    if n == 1:
        return 0
    util = []
    for a, ca in enumerate(candidates):
        s = sum(_CHRF.sentence_score(ca, [cb]).score
                for b, cb in enumerate(candidates) if b != a) / (n - 1)
        util.append(s)
    return argmax(util)


def _spearman(a: list, b: list) -> float:
    """Spearman rank correlation (no tie correction; diagnostic use)."""
    def ranks(xs: list) -> list:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0] * len(xs)
        for rank, i in enumerate(order):
            out[i] = rank
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0


def build_variant(variant: str, rows: list, srcs: list, qes: list,
                  alpha: float, refs: list = None) -> list:
    """Hypotheses (one per source) selected by `variant`."""
    hyps = []
    for r, src, qe in zip(rows, srcs, qes):
        cands = r["candidates"]
        if variant == "oracle":
            idx = oracle_pick(cands, refs[r["idx"]])
        elif variant == "mbr":
            idx = mbr_pick(cands)
        else:
            idx = pick(variant, candidate_signals(src, cands, qe), alpha)
        hyps.append(cands[idx])
    return hyps


def diagnostics(split: str, limit: int) -> dict:
    """Why reranking fails to capture the oracle headroom: per-source QE-vs-reference
    rank correlation, QE/oracle selection agreement, and where the headroom lives."""
    import statistics
    rows, srcs, refs, qes = load_aligned(split, limit)
    n = len(rows)
    qe_top = oracle_top = agree = 0
    spreads, corrs = [], []
    for r, qe in zip(rows, qes):
        cands = r["candidates"]
        ref_chrf = [_CHRF.sentence_score(c, [refs[r["idx"]]]).score for c in cands]
        qp, op = argmax(qe), argmax(ref_chrf)
        qe_top += qp == 0
        oracle_top += op == 0
        agree += qp == op
        spreads.append(round(max(qe) - min(qe), 6))
        corrs.append(_spearman(qe, ref_chrf))
    return {
        "split": split, "n": n, "n_best": len(rows[0]["candidates"]),
        "qe_picks_top_beam_pct": round(qe_top / n * 100, 1),
        "oracle_picks_top_beam_pct": round(oracle_top / n * 100, 1),
        "headroom_in_nontop_beams_pct": round((1 - oracle_top / n) * 100, 1),
        "qe_oracle_agreement_pct": round(agree / n * 100, 1),
        "random_agreement_pct": round(100 / len(rows[0]["candidates"]), 1),
        "mean_within_source_qe_spread": round(statistics.mean(spreads), 4),
        "mean_qe_vs_refchrf_spearman": round(statistics.mean(corrs), 3),
    }


def corpus_chrf(hyps: list, refs: list) -> float:
    """Corpus chrF++ of a hypothesis list against references."""
    return round(_CHRF.corpus_score(hyps, [refs]).score, 2)


def load_aligned(split: str, limit: int) -> tuple:
    """Load (rows, srcs, refs, qes) with N-best and QE aligned to the split."""
    suffix = f"_n{limit}" if limit else ""
    rows = read_jsonl(os.path.join(NBEST, f"{split}{suffix}_nbest.jsonl"))
    qe_rows = {q["idx"]: q["qe"] for q in
               read_jsonl(os.path.join(NBEST, f"{split}{suffix}_qe.jsonl"))}
    srcs, refs = read_split(split, limit)
    assert len(rows) == len(srcs), f"{len(rows)} nbest vs {len(srcs)} sources"
    qes = [qe_rows[r["idx"]] for r in rows]
    return rows, srcs, refs, qes


def tune(split: str, limit: int) -> dict:
    """Grid-search alpha on `split` by corpus chrF++; return the tuning report."""
    rows, srcs, refs, qes = load_aligned(split, limit)
    curve = {}
    for alpha in ALPHA_GRID:
        hyps = build_variant("fused", rows, srcs, qes, alpha)
        curve[alpha] = corpus_chrf(hyps, refs)
    best_alpha = max(curve, key=curve.get)
    return {"split": split, "n": len(srcs), "best_alpha": best_alpha,
            "chrf_by_alpha": curve}


def write_variants(split: str, limit: int, alpha: float) -> dict:
    """Write hyps files for all variants on `split`; return a chrF++ summary."""
    rows, srcs, refs, qes = load_aligned(split, limit)
    suffix = f"_n{limit}" if limit else ""
    summary = {"split": split, "n": len(srcs), "alpha": alpha, "chrf": {}}
    for variant in ("beam1", "qe", "ifs", "fused", "mbr", "oracle"):
        hyps = build_variant(variant, rows, srcs, qes, alpha, refs=refs)
        out = os.path.join(RESULTS, f"rerank_{variant}_{split}{suffix}_hyps.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(hyps))
        summary["chrf"][variant] = corpus_chrf(hyps, refs)
        print(f"  rerank_{variant}: chrF++ {summary['chrf'][variant]} -> {out}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--alpha", type=float, default=0.3, help="fused weight on IFS")
    ap.add_argument("--limit", type=int, default=0, help="0 = full; else _n{limit} files")
    ap.add_argument("--tune", action="store_true", help="grid-search alpha on --split")
    ap.add_argument("--report", default=os.path.join(RESULTS, "rerank_report.json"))
    args = ap.parse_args()

    report = json.load(open(args.report, encoding="utf-8")) \
        if os.path.exists(args.report) else {}

    if args.tune:
        res = tune(args.split, args.limit)
        report["tuning"] = res
        print(f"best alpha on {args.split}: {res['best_alpha']} "
              f"(chrF++ {res['chrf_by_alpha'][res['best_alpha']]})")
        print("curve:", res["chrf_by_alpha"])
    else:
        res = write_variants(args.split, args.limit, args.alpha)
        report[f"variants_{args.split}"] = res
        diag = diagnostics(args.split, args.limit)
        report[f"diagnostics_{args.split}"] = diag
        print(f"diagnostics {args.split}: QE-vs-refchrF++ spearman "
              f"{diag['mean_qe_vs_refchrf_spearman']} | QE/oracle agree "
              f"{diag['qe_oracle_agreement_pct']}% (random {diag['random_agreement_pct']}%) | "
              f"headroom in non-top beams {diag['headroom_in_nontop_beams_pct']}%")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"-> {args.report}")


if __name__ == "__main__":
    main()
