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


DEFAULT_COMET = "Unbabel/wmt22-comet-da"


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


def _comet(srcs, hyps, refs, model_name):
    """Reference-based COMET (e.g. wmt22-comet-da). Needs source, hyp, ref.

    Returns (system_score, model_name) or (None, note) on failure so the caller
    can record why it was skipped instead of crashing the whole eval.
    """
    if srcs is None:
        return None, "comet skipped: no source (pass --refs jsonl with an 'en' field)"
    try:
        from comet import download_model, load_from_checkpoint
    except Exception:
        return None, "comet not installed; pip install unbabel-comet"
    try:
        ckpt = download_model(model_name)
        model = load_from_checkpoint(ckpt)
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(srcs, hyps, refs)]
        import torch
        gpus = 1 if torch.cuda.is_available() else 0
        out = model.predict(data, batch_size=16, gpus=gpus)
        return round(float(out["system_score"]), 4), model_name
    except Exception as exc:  # surface the reason, don't kill chrF/BLEU
        return None, f"comet error: {type(exc).__name__}: {exc}"


def score(hyps: list, refs: list, system: str,
          srcs: list = None, run_comet: bool = False,
          comet_model: str = DEFAULT_COMET) -> dict:
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

    if run_comet:
        comet_score, note = _comet(srcs, hyps, refs, comet_model)
        out["comet"] = comet_score
        out["comet_model"] = comet_model if comet_score is not None else None
        out["comet_note"] = note if comet_score is None else "ok"
    else:
        out["comet"] = None
        out["comet_note"] = "comet not run; pass --comet (run on GPU/Colab)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--field", default="my")
    ap.add_argument("--src-field", default="en", help="source field for COMET")
    ap.add_argument("--system", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N refs")
    ap.add_argument("--comet", action="store_true", help="compute COMET (needs GPU)")
    ap.add_argument("--comet-model", default=DEFAULT_COMET)
    ap.add_argument("--out", default=os.path.join(RESULTS, "main_results.json"))
    args = ap.parse_args()

    hyps = read_lines(args.hyps)
    refs = read_refs(args.refs, args.field)
    srcs = read_refs(args.refs, args.src_field) if args.comet else None
    if args.limit:
        refs = refs[: args.limit]
        if srcs is not None:
            srcs = srcs[: args.limit]
    res = score(hyps, refs, args.system, srcs=srcs,
                run_comet=args.comet, comet_model=args.comet_model)
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
