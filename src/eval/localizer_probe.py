"""Flagship smoke test: is IFS useful as a span-level *localizer* even though its
scalar fails (RQ3)?

The human study collected per-item yes/no judgments on whether the translation kept
the right quantities, entities, and step order. IFS exposes matching automatic
components (quantity, entity, step). This probe asks, per component: does the IFS
component score separate the human "yes" cases from the "no" cases?

Decisive subtlety: IFS components return 1.0 trivially when the source has no
numeral/entity. We therefore report discrimination on the *informative* subset
(source actually contains the feature), which is the only place a localizer could
help. Discrimination = point-biserial r + AUC of (component score) vs (human yes=1).

Note: IFS has NO automatic action component (the source docstring says action is
human-only), so action_correct cannot be probed -- itself a finding.

Usage:
  python src/eval/localizer_probe.py --ratings experiments/results/ratings_filled.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import ifs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "experiments", "results")

# component -> (human yes/no column, function testing if the SOURCE is informative)
COMPONENTS = {
    "quantity": ("quantities_correct", lambda src: bool(ifs.extract_numerals(src))),
    "entity": ("entities_correct", lambda src: bool(ifs.extract_entities(src))),
    "step": ("step_order", lambda src: True),
}
YES, NO = "yes", "no"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a positive (label=1) outranks a negative; rank-based AUC."""
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_mean = np.array([ranks[scores == s].mean() for s in scores])
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (tie_mean[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def discriminate(scores: list, labels: list) -> dict:
    """Point-biserial r, AUC, and per-class means for a component vs human yes/no."""
    s, y = np.array(scores, dtype=float), np.array(labels, dtype=int)
    n_yes, n_no = int((y == 1).sum()), int((y == 0).sum())
    out = {"n": len(y), "n_yes": n_yes, "n_no": n_no,
           "mean_when_yes": round(float(s[y == 1].mean()), 4) if n_yes else None,
           "mean_when_no": round(float(s[y == 0].mean()), 4) if n_no else None}
    if n_yes and n_no and s.std() > 0:
        r = stats.pointbiserialr(y, s)
        out["pointbiserial_r"] = round(float(r.statistic), 4)
        out["p"] = round(float(r.pvalue), 5)
        out["auc"] = round(float(auc(s, y)), 4)
    else:
        out["pointbiserial_r"] = None
        out["p"] = None
        out["auc"] = None
    return out


def load(ratings_path: str) -> list:
    """Rated rows with the human yes/no columns (no key needed; IFS is source-anchored)."""
    with open(ratings_path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("followability", "").strip()]


def probe(rows: list) -> dict:
    """Per component, run rating-level and item-level (majority-vote) discrimination,
    on all rows and on the informative subset (source has the feature)."""
    # cache IFS components per unique (src, hyp)
    comp_cache = {}
    for r in rows:
        keyp = (r["src_en"], r["hyp_my"])
        if keyp not in comp_cache:
            comp_cache[keyp] = ifs.segment_ifs(r["src_en"], r["hyp_my"])

    result = {}
    for comp, (col, informative) in COMPONENTS.items():
        rating_all, rating_inf = ([], []), ([], [])
        item_human = defaultdict(list)
        item_score, item_inf = {}, {}
        for r in rows:
            val = r.get(col, "").strip().lower()
            if val not in (YES, NO):
                continue
            score = comp_cache[(r["src_en"], r["hyp_my"])][comp]
            label = 1 if val == YES else 0
            is_inf = informative(r["src_en"])
            rating_all[0].append(score)
            rating_all[1].append(label)
            if is_inf:
                rating_inf[0].append(score)
                rating_inf[1].append(label)
            item_human[r["id"]].append(label)
            item_score[r["id"]] = score
            item_inf[r["id"]] = is_inf

        ids = list(item_human)
        item_scores = [item_score[i] for i in ids]
        item_labels = [1 if np.mean(item_human[i]) >= 0.5 else 0 for i in ids]
        inf_ids = [k for k, i in enumerate(ids) if item_inf[ids[k]]]
        result[comp] = {
            "rating_level_all": discriminate(*rating_all),
            "rating_level_informative": discriminate(*rating_inf),
            "item_level_all": discriminate(item_scores, item_labels),
            "item_level_informative": discriminate(
                [item_scores[k] for k in inf_ids], [item_labels[k] for k in inf_ids]),
        }
    result["action"] = {"note": "IFS has no automatic action component; action_correct "
                                "cannot be probed -- a structural gap in the metric."}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS, "localizer_probe.json"))
    args = ap.parse_args()
    rows = load(args.ratings)
    result = {"n_ratings": len(rows), "components": probe(rows)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
