"""Automatic MT metrics for the WikiHow-MY benchmark.

Primary metric is chrF++ (segmentation-agnostic, the right choice for unsegmented
Myanmar). We also report spBLEU (FLORES-200 SentencePiece tokenizer, for
comparability with NLLB/FLORES results) and tokenized BLEU. COMET is computed if
`comet` is installed (lazy); otherwise skipped with a note. Results are written to
experiments/results/<...>.json keyed by system so paper tables are generated, never
hand-typed.

Usage:
  python src/eval/automatic.py --hyps experiments/results/nllb_zeroshot_test_hyps.txt \
      --refs data/processed/test.jsonl --system nllb_zeroshot
"""
import argparse
import json
import os

import sacrebleu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "experiments", "results")


def read_refs(path: str, field: str = "my") -> list:
    refs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            refs.append(json.loads(line)[field])
    return refs


def read_lines(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _spbleu(hyps, refs):
    """spBLEU = BLEU with the FLORES-200 SPM tokenizer; fall back across names."""
    for tok in ("flores200", "flores101", "spm"):
        try:
            m = sacrebleu.BLEU(tokenize=tok)
            s = m.corpus_score(hyps, [refs])
            return s.score, m.get_signature().format(), tok
        except Exception:
            continue
    return None, None, None


def score(hyps: list, refs: list, system: str) -> dict:
    assert len(hyps) == len(refs), f"{len(hyps)} hyps vs {len(refs)} refs"
    chrf = sacrebleu.CHRF(word_order=2)  # chrF++
    chrf_s = chrf.corpus_score(hyps, [refs])
    bleu = sacrebleu.BLEU()
    bleu_s = bleu.corpus_score(hyps, [refs])
    sp, sp_sig, sp_tok = _spbleu(hyps, refs)

    out = {
        "system": system,
        "n": len(hyps),
        "chrf++": round(chrf_s.score, 2),
        "chrf++_signature": chrf.get_signature().format(),
        "bleu": round(bleu_s.score, 2),
        "bleu_signature": bleu.get_signature().format(),
        "spbleu": round(sp, 2) if sp is not None else None,
        "spbleu_tokenizer": sp_tok,
        "spbleu_signature": sp_sig,
    }

    try:  # COMET is optional / heavy
        from comet import download_model, load_from_checkpoint  # noqa
        # left as a hook; run on GPU/Colab where the checkpoint is cached
        out["comet"] = None
        out["comet_note"] = "comet installed but not run here; compute on GPU"
    except Exception:
        out["comet"] = None
        out["comet_note"] = "comet not installed; compute on Colab/GPU"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--field", default="my")
    ap.add_argument("--system", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N refs")
    ap.add_argument("--out", default=os.path.join(RESULTS, "main_results.json"))
    args = ap.parse_args()

    hyps = read_lines(args.hyps)
    refs = read_refs(args.refs, args.field)
    if args.limit:
        refs = refs[: args.limit]
    res = score(hyps, refs, args.system)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    allres = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            allres = json.load(f)
    allres[args.system] = res
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(allres, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
