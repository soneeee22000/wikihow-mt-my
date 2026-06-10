"""Build MetricX-24 reference-based input JSONLs from hypotheses + references.

MetricX-24's predict.py expects one JSON object per line with the fields
"source", "hypothesis", "reference" (reference-based mode). We emit one input
file per system into experiments/results/metricx_inputs/, which is then bundled
into the Kaggle dataset and scored on GPU by scripts/kaggle/run_metricx.py.

Run locally: python src/eval/build_metricx_inputs.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "metricx_inputs")

# system -> (reference split jsonl, hypotheses file)
SYSTEMS = {
    "nllb_zeroshot": ("test.jsonl", "nllb_zeroshot_test_hyps.txt"),
    "nllb_finetuned": ("test.jsonl", "nllb_finetuned_test_hyps.txt"),
    "gemini": ("test.jsonl", "gemini_test_hyps.txt"),
    "gtranslate": ("test.jsonl", "gtranslate_test_hyps.txt"),
    "nllb_zeroshot_flores": ("flores.jsonl", "nllb_zeroshot_flores_hyps.txt"),
    "nllb_finetuned_flores": ("flores.jsonl", "nllb_finetuned_flores_hyps.txt"),
    "gtranslate_flores": ("flores.jsonl", "gtranslate_flores_hyps.txt"),
    "gemini_flores": ("flores.jsonl", "gemini_flores_hyps.txt"),
}


def read_split(name: str) -> tuple:
    """Return (sources, references) lists from a processed jsonl split."""
    srcs, refs = [], []
    with open(os.path.join(PROC, name), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            srcs.append(row["en"])
            refs.append(row["my"])
    return srcs, refs


def read_hyps(name: str) -> list:
    """Return hypothesis lines (one translation per line)."""
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def main() -> None:
    """Emit one MetricX input JSONL per system."""
    os.makedirs(OUT, exist_ok=True)
    for system, (split, hyp_file) in SYSTEMS.items():
        srcs, refs = read_split(split)
        hyps = read_hyps(hyp_file)
        assert len(srcs) == len(hyps) == len(refs), (
            f"{system}: len mismatch src={len(srcs)} hyp={len(hyps)} ref={len(refs)}")
        path = os.path.join(OUT, f"{system}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s, h, r in zip(srcs, hyps, refs):
                f.write(json.dumps({"source": s, "hypothesis": h, "reference": r},
                                   ensure_ascii=False) + "\n")
        print(f"wrote {len(hyps)} -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
