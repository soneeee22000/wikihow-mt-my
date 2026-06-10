"""Standalone COMET scorer — run from an ISOLATED venv so unbabel-comet's heavy
deps never touch the main project env.

Computes reference-based COMET (Unbabel/wmt22-comet-da) from already-produced
hypotheses and merges the score into experiments/results/main_results.json WITHOUT
recomputing or overwriting the other metrics. Only needs `unbabel-comet` (+torch).

Usage (from the venv's python):
  <venv>/Scripts/python.exe scripts/run_comet.py --systems nllb_zeroshot,nllb_finetuned
  add --limit 4 for a smoke check first.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "experiments", "results")
MAIN = os.path.join(RES, "main_results.json")
MODEL = "Unbabel/wmt22-comet-da"


def read_field(path: str, field: str) -> list:
    """Read one field (e.g. 'en' / 'my') from a jsonl file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[field] for line in f]


def read_lines(path: str) -> list:
    """Read hypotheses, one per line."""
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def comet_score(srcs: list, hyps: list, refs: list, model_name: str) -> float:
    """Reference-based COMET system score (downloads the model on first use)."""
    from comet import download_model, load_from_checkpoint
    import torch
    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    gpus = 1 if torch.cuda.is_available() else 0
    out = model.predict(data, batch_size=16, gpus=gpus)
    return round(float(out["system_score"]), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True, help="comma-separated system names")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = full; else first N (smoke)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    srcs = read_field(os.path.join(PROC, f"{args.split}.jsonl"), "en")
    refs = read_field(os.path.join(PROC, f"{args.split}.jsonl"), "my")
    if args.limit:
        srcs, refs = srcs[: args.limit], refs[: args.limit]

    results = json.load(open(MAIN, encoding="utf-8")) if os.path.exists(MAIN) else {}
    for system in args.systems.split(","):
        hyps = read_lines(os.path.join(RES, f"{system}_{args.split}_hyps.txt"))
        if args.limit:
            hyps = hyps[: args.limit]
        score = comet_score(srcs, hyps, refs, args.model)
        print(f"{system}: COMET {score}", flush=True)
        if args.limit:  # smoke: don't pollute the real results file
            continue
        results.setdefault(system, {})["comet"] = score
        results[system]["comet_model"] = args.model

    if not args.limit:
        with open(MAIN, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"merged COMET -> {MAIN}", flush=True)


if __name__ == "__main__":
    main()
