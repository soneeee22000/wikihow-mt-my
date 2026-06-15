"""Flagship Phase 3 smoke probe: can a learned (LLM-judge) per-dimension faithfulness
estimator predict human followability where the surface IFS metric fails (RQ3, r~0)?

For each rated item we ask an LLM judge (Gemini), given the English source and the
Burmese translation, to judge four procedural-faithfulness dimensions yes/no:
step order, action/verb, entities, quantities. We then test two claims:
  (A) Recovery -- does the judge agree with the human per-dimension yes/no labels?
      (accuracy + phi correlation), especially `action_correct`, the keystone the
      surface IFS metric has no component for.
  (B) Usefulness -- do the judge's per-dimension scores predict human *followability*
      (Pearson/Spearman) where IFS's item-level r is ~0.11? A composite (mean of the
      four dims) and the action dimension alone.

Smoke-first: --limit N validates the whole pipeline (API -> parse -> score ->
correlate) on a few items in ~1 min before the full run. Judge outputs are cached by
opaque item id (no source text in the cache) so re-runs never re-hit the API.

Usage:
  python src/eval/llm_judge_probe.py --ratings experiments/results/ratings_filled.csv --limit 8
  python src/eval/llm_judge_probe.py --ratings experiments/results/ratings_filled.csv
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import requests
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "experiments", "results")
DEFAULT_CACHE = os.path.join(RESULTS, "llm_judge_cache.json")

DIMENSIONS = ["step_order", "action_correct", "entities_correct", "quantities_correct"]
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
YES, NO = "yes", "no"

JUDGE_INSTRUCTIONS = (
    "You are evaluating a Burmese (Myanmar) translation of an English how-to "
    "instruction for PROCEDURAL FAITHFULNESS -- whether a reader could correctly "
    "carry out the procedure from the Burmese alone. Judge each dimension \"yes\" or "
    "\"no\":\n"
    "- step_order: steps in the same order, none merged, dropped, or reordered.\n"
    "- action_correct: the reader is told to do the RIGHT action (imperative verbs "
    "preserved, no negation flips).\n"
    "- entities_correct: named objects/tools/ingredients preserved.\n"
    "- quantities_correct: numbers, amounts, units, temperatures preserved.\n"
    "If a dimension does not apply (e.g. the source has no quantities), answer \"yes\".\n"
    "Return ONLY a JSON object with exactly these keys and \"yes\"/\"no\" values."
)


def load_dotenv() -> None:
    """Load KEY=VALUE lines from the gitignored project-root .env into os.environ."""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_items(ratings_path: str) -> dict:
    """Aggregate the blind ratings CSV per item: source, hypothesis, human
    followability mean, and human per-dimension majority labels (1=yes)."""
    by_id = defaultdict(list)
    with open(ratings_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("followability", "").strip():
                by_id[r["id"]].append(r)
    items = {}
    for item_id, rows in by_id.items():
        human_dims = {}
        for dim in DIMENSIONS:
            labels = [1 if r.get(dim, "").strip().lower() == YES else 0
                      for r in rows if r.get(dim, "").strip().lower() in (YES, NO)]
            human_dims[dim] = int(np.mean(labels) >= 0.5) if labels else None
        items[item_id] = {
            "src_en": rows[0]["src_en"],
            "hyp_my": rows[0]["hyp_my"],
            "followability": float(np.mean([float(r["followability"]) for r in rows])),
            "human_dims": human_dims,
        }
    return items


def call_judge(src: str, hyp: str, model: str, api_key: str) -> dict:
    """One Gemini call returning {dimension: "yes"|"no"} via JSON response mode."""
    prompt = f"{JUDGE_INSTRUCTIONS}\n\nEnglish source:\n{src}\n\nBurmese translation:\n{hyp}"
    schema = {"type": "object", "properties": {d: {"type": "string", "enum": [YES, NO]}
                                               for d in DIMENSIONS}, "required": DIMENSIONS}
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": schema}}
    url = GEMINI_URL.format(model=model)
    for attempt in range(MAX_RETRIES):
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            return {d: parsed.get(d, "").strip().lower() for d in DIMENSIONS}
        if resp.status_code in (429, 500, 503) and attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError("judge call failed after retries")


def judge_items(items: dict, model: str, cache_path: str, limit: int) -> dict:
    """Run (or load cached) judge labels for up to `limit` items, persisting after each."""
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    ids = sorted(items)[:limit] if limit else sorted(items)
    for n, item_id in enumerate(ids, 1):
        if item_id in cache:
            continue
        if not api_key:
            raise SystemExit("GEMINI_API_KEY not set (.env) -- needed to run the judge.")
        it = items[item_id]
        cache[item_id] = call_judge(it["src_en"], it["hyp_my"], model, api_key)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if n % 10 == 0 or n == len(ids):
            print(f"  judged {n}/{len(ids)}")
    return {i: cache[i] for i in ids if i in cache}


def phi(judge: list, human: list) -> float:
    """Phi coefficient (Pearson of two binary vectors); nan if either is constant."""
    j, h = np.array(judge, float), np.array(human, float)
    if j.std() == 0 or h.std() == 0:
        return float("nan")
    return round(float(stats.pearsonr(j, h).statistic), 4)


def recovery(items: dict, judged: dict) -> dict:
    """Claim A: per-dimension judge-vs-human agreement (accuracy + phi)."""
    out = {}
    for dim in DIMENSIONS:
        pairs = [(1 if judged[i].get(dim) == YES else 0, items[i]["human_dims"][dim])
                 for i in judged if items[i]["human_dims"][dim] is not None]
        if not pairs:
            continue
        jl, hl = [p[0] for p in pairs], [p[1] for p in pairs]
        out[dim] = {"n": len(pairs), "accuracy": round(float(np.mean(np.array(jl) == hl)), 4),
                    "human_yes_rate": round(float(np.mean(hl)), 4),
                    "judge_yes_rate": round(float(np.mean(jl)), 4), "phi": phi(jl, hl)}
    return out


def usefulness(items: dict, judged: dict) -> dict:
    """Claim B: do judge scores predict human followability where IFS gave ~0?"""
    ids = list(judged)
    follow = [items[i]["followability"] for i in ids]
    scores = {"judge_composite": [np.mean([1 if judged[i].get(d) == YES else 0
                                           for d in DIMENSIONS]) for i in ids]}
    for dim in DIMENSIONS:
        scores[f"judge_{dim}"] = [1 if judged[i].get(dim) == YES else 0 for i in ids]
    out = {"n_items": len(ids), "vs_followability": {}}
    for name, vals in scores.items():
        if np.std(vals) == 0:
            out["vs_followability"][name] = {"pearson": None, "note": "constant"}
            continue
        pr, sr = stats.pearsonr(vals, follow), stats.spearmanr(vals, follow)
        out["vs_followability"][name] = {
            "pearson": round(float(pr.statistic), 4), "pearson_p": round(float(pr.pvalue), 5),
            "spearman": round(float(sr.statistic), 4), "spearman_p": round(float(sr.pvalue), 5)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--model", default=GEMINI_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="judge only first N items (smoke)")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=os.path.join(RESULTS, "llm_judge_probe.json"))
    args = ap.parse_args()

    items = build_items(args.ratings)
    print(f"items with human labels: {len(items)}; judging "
          f"{args.limit or len(items)} with {args.model}")
    judged = judge_items(items, args.model, args.cache, args.limit)
    result = {"model": args.model, "n_items_judged": len(judged),
              "recovery_judge_vs_human": recovery(items, judged),
              "usefulness_vs_followability": usefulness(items, judged)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
