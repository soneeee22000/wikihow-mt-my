"""One-off COMET scorer for the FLORES+ cross-eval rows.

Mirrors scripts/run_comet.py but keeps the FLORES result keys distinct
(nllb_zeroshot_flores / nllb_finetuned_flores) so the WikiHow COMET scores on
nllb_zeroshot / nllb_finetuned are never overwritten. Run from the isolated
COMET venv: C:\\comet_venv\\Scripts\\python.exe scripts/run_comet_flores.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "experiments", "results")
MAIN = os.path.join(RES, "main_results.json")
MODEL = "Unbabel/wmt22-comet-da"
REF = os.path.join(PROC, "flores.jsonl")
SYSTEMS = {
    "nllb_zeroshot_flores": "nllb_zeroshot_flores_hyps.txt",
    "nllb_finetuned_flores": "nllb_finetuned_flores_hyps.txt",
    "gtranslate_flores": "gtranslate_flores_hyps.txt",
    "gemini_flores": "gemini_flores_hyps.txt",
}


def field(path: str, name: str) -> list:
    """Read one field from a jsonl file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[name] for line in f]


def main() -> None:
    """Score each FLORES system and merge COMET into the matching result key."""
    from comet import download_model, load_from_checkpoint
    import torch

    srcs, refs = field(REF, "en"), field(REF, "my")
    model = load_from_checkpoint(download_model(MODEL))
    gpus = 1 if torch.cuda.is_available() else 0
    results = json.load(open(MAIN, encoding="utf-8"))

    for system, hyp_file in SYSTEMS.items():
        path = os.path.join(RES, hyp_file)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"{system}: hyp file missing or empty, skipping", flush=True)
            continue
        if results.get(system, {}).get("comet") is not None:
            print(f"{system}: COMET already present, skipping", flush=True)
            continue
        with open(path, encoding="utf-8") as f:
            hyps = [ln.rstrip("\n") for ln in f]
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        score = round(float(model.predict(data, batch_size=16, gpus=gpus)["system_score"]), 4)
        print(f"{system}: COMET {score}", flush=True)
        results.setdefault(system, {})["comet"] = score
        results[system]["comet_model"] = MODEL
        results[system]["comet_note"] = "ok"

    with open(MAIN, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"merged FLORES COMET -> {MAIN}", flush=True)


if __name__ == "__main__":
    main()
