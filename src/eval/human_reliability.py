"""Human-study reliability + item-level metric validation (RQ3, paper §6).

Two analyses the rating-level correlate.py cannot give:
  1. Krippendorff's alpha (ordinal) on followability/adequacy over every item rated
     by >=2 annotators -- establishes the human study is reliable enough to trust.
  2. Item-level correlation: collapse each item to its mean human followability,
     then correlate against IFS/chrF/BLEU. This removes the pseudo-replication of
     counting the 6-7 overlap ratings as independent observations.

Usage:
  python src/eval/human_reliability.py \
      --ratings experiments/results/ratings_filled.csv \
      --key experiments/results/ratings_key.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import sacrebleu
from scipy import stats

import ifs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "experiments", "results")
MIN_RATERS_FOR_ALPHA = 2


def ordinal_krippendorff_alpha(ratings_by_unit: dict) -> dict:
    """Krippendorff's alpha for ordinal data.

    ratings_by_unit maps unit id -> list of integer ratings (units with <2 ratings
    are ignored, per the standard definition). Uses the coincidence-matrix form with
    the ordinal difference metric.
    """
    units = {u: vs for u, vs in ratings_by_unit.items() if len(vs) >= MIN_RATERS_FOR_ALPHA}
    if not units:
        return {"alpha": None, "n_units": 0, "n_pairable_values": 0}

    values = sorted({v for vs in units.values() for v in vs})
    index = {v: i for i, v in enumerate(values)}
    size = len(values)
    coincidence = np.zeros((size, size), dtype=float)
    for vs in units.values():
        weight = 1.0 / (len(vs) - 1)
        for a, b in combinations(vs, 2):
            ia, ib = index[a], index[b]
            coincidence[ia, ib] += weight
            coincidence[ib, ia] += weight

    marginals = coincidence.sum(axis=1)
    total = marginals.sum()
    if total <= 1:
        return {"alpha": None, "n_units": len(units), "n_pairable_values": int(total)}

    def ordinal_delta_sq(i: int, j: int) -> float:
        """Squared ordinal distance between value-indices i and j."""
        lo, hi = (i, j) if i <= j else (j, i)
        span = marginals[lo:hi + 1].sum() - (marginals[lo] + marginals[hi]) / 2.0
        return float(span ** 2)

    observed = 0.0
    expected = 0.0
    for i in range(size):
        for j in range(i + 1, size):
            delta = ordinal_delta_sq(i, j)
            observed += coincidence[i, j] * delta
            expected += marginals[i] * marginals[j] * delta
    expected /= (total - 1)
    if expected == 0:
        return {"alpha": None, "n_units": len(units), "n_pairable_values": int(total)}
    alpha = 1.0 - observed / expected
    return {"alpha": round(alpha, 4), "n_units": len(units),
            "n_pairable_values": int(total), "scale_points": values}


def load(ratings_path: str, key_path: str) -> list:
    """Join filled ratings to the private key, keeping only rated rows."""
    with open(key_path, encoding="utf-8") as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    rows = []
    with open(ratings_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("followability", "").strip():
                continue
            k = key.get(r["id"], {})
            rows.append({**r, "ref_my": k.get("ref_my", ""), "system": k.get("system", "")})
    return rows


def item_level(rows: list) -> dict:
    """Mean human followability per item vs item-level automatic metrics."""
    fby = defaultdict(list)
    meta = {}
    for r in rows:
        fby[r["id"]].append(float(r["followability"]))
        meta[r["id"]] = r
    ids = sorted(fby)
    human = np.array([float(np.mean(fby[i])) for i in ids])
    scored = {"ifs": [], "chrf": [], "bleu": []}
    for i in ids:
        m = meta[i]
        scored["ifs"].append(ifs.segment_ifs(m["src_en"], m["hyp_my"])["ifs"] * 100)
        scored["chrf"].append(sacrebleu.sentence_chrf(m["hyp_my"], [m["ref_my"]], word_order=2).score)
        scored["bleu"].append(sacrebleu.sentence_bleu(m["hyp_my"], [m["ref_my"]]).score)
    corr = {}
    for name, vals in scored.items():
        arr = np.array(vals)
        pr = stats.pearsonr(arr, human)
        sr = stats.spearmanr(arr, human)
        corr[name] = {"pearson": round(float(pr.statistic), 4), "pearson_p": round(float(pr.pvalue), 5),
                      "spearman": round(float(sr.statistic), 4), "spearman_p": round(float(sr.pvalue), 5),
                      "mean": round(float(arr.mean()), 2), "sd": round(float(arr.std(ddof=1)), 2),
                      "min": round(float(arr.min()), 2), "max": round(float(arr.max()), 2)}
    return {"n_items": len(ids), "human_followability_mean": round(float(human.mean()), 3),
            "human_followability_sd": round(float(human.std(ddof=1)), 3), "metrics": corr}


def per_system(rows: list) -> dict:
    """Mean human followability and mean IFS per system, aggregated over items.

    Each item contributes once (its mean rating), not once per rating. The 30-item
    overlap block carries up to 9 ratings and is unbalanced across systems, so a
    rating-weighted mean would over-weight it; item-level means keep the 40 items
    per system balanced.
    """
    out = {}
    by_item = defaultdict(list)
    meta = {}
    for r in rows:
        by_item[r["id"]].append(float(r["followability"]))
        meta[r["id"]] = r
    by_sys = defaultdict(list)
    for item_id, vals in by_item.items():
        by_sys[meta[item_id]["system"]].append(item_id)
    for sys_name, ids in sorted(by_sys.items()):
        f = np.array([float(np.mean(by_item[i])) for i in ids])
        s = np.array([ifs.segment_ifs(meta[i]["src_en"], meta[i]["hyp_my"])["ifs"] * 100 for i in ids])
        out[sys_name] = {"n": len(ids), "n_ratings": sum(len(by_item[i]) for i in ids),
                         "human_mean": round(float(f.mean()), 2), "human_sd": round(float(f.std(ddof=1)), 2),
                         "ifs_mean": round(float(s.mean()), 2), "ifs_sd": round(float(s.std(ddof=1)), 2)}
    return out


def system_ranking(per_sys: dict) -> dict:
    """Descriptive Spearman of per-system IFS means against human means (n=4 systems).

    Reported without a p-value: n=4 is far too small for an inferential claim
    \\citep{mathur2020tangled}.
    """
    names = sorted(per_sys)
    human = [per_sys[n]["human_mean"] for n in names]
    ifs_v = [per_sys[n]["ifs_mean"] for n in names]
    return {"n_systems": len(names),
            "spearman_human_ifs": round(float(stats.spearmanr(ifs_v, human).statistic), 3),
            "ifs_spread": round(max(ifs_v) - min(ifs_v), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--key", default=os.path.join(RESULTS, "ratings_key.csv"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "human_reliability.json"))
    args = ap.parse_args()

    rows = load(args.ratings, args.key)
    follow_by_unit = defaultdict(list)
    adeq_by_unit = defaultdict(list)
    for r in rows:
        follow_by_unit[r["id"]].append(int(float(r["followability"])))
        if r.get("adequacy", "").strip():
            adeq_by_unit[r["id"]].append(int(float(r["adequacy"])))

    result = {
        "n_ratings": len(rows),
        "n_raters": len({r["annotator"] for r in rows}),
        "reliability": {
            "followability_alpha_ordinal": ordinal_krippendorff_alpha(follow_by_unit),
            "adequacy_alpha_ordinal": ordinal_krippendorff_alpha(adeq_by_unit),
        },
        "item_level": item_level(rows),
        "per_system": per_system(rows),
    }
    result["system_ranking"] = system_ranking(result["per_system"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
