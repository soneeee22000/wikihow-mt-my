"""Flagship hardening: does the learned per-dimension estimator recover what IFS loses?

The floor paper's headline negative result is that the surface IFS metric INVERTS the
system ranking (system-level Spearman(human, IFS) = -0.80) and tracks item-level human
followability at only r~0.11. This script tests the positive counterpart on data already
on disk (the cached LLM-judge labels -- NO new API calls):

  (1) Ranking recovery -- per-system mean of the judge composite vs the human ranking,
      reported next to IFS's inversion. The "...and here is what works" punchline.
  (2) Item-level usefulness -- judge composite vs human followability (Pearson + bootstrap
      CI) and a Williams test for the judge-vs-IFS gap (dependent correlations, shared
      human variable), so 0.54 vs 0.11 is statistically defensible.
  (3) Self-preference control -- (1) and (2) recomputed with the judge model's own
      translations (gemini) held out, so the recovery is not just style self-favouring.

Usage:
  python src/eval/estimator_ranking.py \
      --ratings experiments/results/ratings_filled.csv \
      --key experiments/results/ratings_key.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import correlate  # src/eval on sys.path[0]: williams_test, bootstrap_ci
import ifs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "experiments", "results")
DEFAULT_CACHE = os.path.join(RESULTS, "llm_judge_cache.json")
DIMENSIONS = ["step_order", "action_correct", "entities_correct", "quantities_correct"]
JUDGE_MODEL_SYSTEM = "gemini"  # held out for the self-preference control
YES = "yes"


def build_items(ratings_path: str, key_path: str, cache: dict) -> dict:
    """Join cached judge labels with per-item human followability, system and per-item
    IFS, keeping only items that carry both a human rating and a judge label."""
    with open(key_path, encoding="utf-8") as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    by_id = defaultdict(list)
    with open(ratings_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("followability", "").strip():
                by_id[r["id"]].append(r)
    items = {}
    for item_id, rows in by_id.items():
        if item_id not in cache or item_id not in key:
            continue
        src, hyp = rows[0]["src_en"], rows[0]["hyp_my"]
        composite = float(np.mean([1.0 if cache[item_id].get(d) == YES else 0.0
                                   for d in DIMENSIONS]))
        items[item_id] = {
            "system": key[item_id]["system"],
            "followability": float(np.mean([float(r["followability"]) for r in rows])),
            "ifs": ifs.segment_ifs(src, hyp)["ifs"] * 100,
            "judge_composite": composite,
        }
    return items


def ranking(items: dict) -> dict:
    """Per-system human/IFS/judge means and the system-level Spearman of each automatic
    score against the human ranking (n=4 systems, descriptive -- no p-value)."""
    systems = sorted({it["system"] for it in items.values()})
    per_system, human, ifs_v, judge_v = {}, [], [], []
    for sysname in systems:
        vals = [it for it in items.values() if it["system"] == sysname]
        h = float(np.mean([v["followability"] for v in vals]))
        f = float(np.mean([v["ifs"] for v in vals]))
        j = float(np.mean([v["judge_composite"] for v in vals]))
        per_system[sysname] = {"n_items": len(vals), "human_mean": round(h, 3),
                               "ifs_mean": round(f, 2), "judge_mean": round(j, 3)}
        human.append(h); ifs_v.append(f); judge_v.append(j)
    return {
        "per_system": per_system,
        "spearman_human_ifs": round(float(stats.spearmanr(human, ifs_v).statistic), 3),
        "spearman_human_judge": round(float(stats.spearmanr(human, judge_v).statistic), 3),
        "n_systems": len(systems),
    }


def usefulness(items: dict) -> dict:
    """Item-level judge composite and IFS vs human followability, with a Williams test
    for the judge-vs-IFS difference (both share the human variable)."""
    human = [it["followability"] for it in items.values()]
    judge = [it["judge_composite"] for it in items.values()]
    ifs_vals = [it["ifs"] for it in items.values()]
    judge_corr = correlate.correlate(judge, human)
    ifs_corr = correlate.correlate(ifs_vals, human)
    r_ab = float(np.corrcoef(judge, ifs_vals)[0, 1])
    will = correlate.williams_test(judge_corr["pearson"], ifs_corr["pearson"], r_ab, len(human))
    will["judge_higher"] = judge_corr["pearson"] > ifs_corr["pearson"]
    return {"n_items": len(human), "judge_composite": judge_corr, "ifs": ifs_corr,
            "williams_judge_vs_ifs": will}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--key", default=os.path.join(RESULTS, "ratings_key.csv"))
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=os.path.join(RESULTS, "estimator_ranking.json"))
    args = ap.parse_args()

    cache = json.load(open(args.cache, encoding="utf-8"))
    items = build_items(args.ratings, args.key, cache)
    held_out = {k: v for k, v in items.items() if v["system"] != JUDGE_MODEL_SYSTEM}
    result = {
        "n_items": len(items),
        "ranking": ranking(items),
        "usefulness": usefulness(items),
        "self_preference_control": {
            "held_out_system": JUDGE_MODEL_SYSTEM,
            "n_items": len(held_out),
            "ranking": ranking(held_out),
            "usefulness": usefulness(held_out),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
