"""N-best NLLB generation, en->my, for the reranking booster.

Beam search with num_return_sequences=N, emitting one JSONL row per source:
  {"idx": i, "candidates": ["cand_0", ..., "cand_{N-1}"]}
candidate 0 is the top beam (the standard greedy/beam decode). Output goes to
experiments/results/nbest/<split>_nbest.jsonl, aligned with the split rows.

Runs on GPU (Kaggle, full) or CPU (slow, smoke). Mirrors src/infer/translate.py.

Usage:
  python src/infer/translate_nbest.py --split test --model checkpoints/nllb_finetuned_wikihow --n-best 16
  python src/infer/translate_nbest.py --split test --model <ckpt> --limit 8   # smoke
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
NBEST = os.path.join(ROOT, "experiments", "results", "nbest")
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "mya_Mymr"
SEED = 42


def tgt_bos_id(tokenizer) -> int:
    """forced_bos_token_id for the target language, across transformers versions."""
    tid = tokenizer.convert_tokens_to_ids(TGT_LANG)
    if tid is not None and tid != tokenizer.unk_token_id:
        return tid
    return tokenizer.lang_code_to_id[TGT_LANG]  # older API


def read_sources(split: str, limit: int) -> list:
    """Load en sources from data/processed/<split>.jsonl (first `limit` if >0)."""
    rows = []
    with open(os.path.join(PROC, f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return [r["en"] for r in rows]


def generate_nbest(srcs: list, model, tok, bos: int, device: str,
                   n_best: int, batch_size: int, max_new_tokens: int) -> list:
    """Return a list (per source) of n_best decoded candidates (top beam first)."""
    out = []
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_new_tokens).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 num_beams=n_best, num_return_sequences=n_best,
                                 max_new_tokens=max_new_tokens)
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch)):  # generate returns n_best rows per input, in order
            out.append(decoded[j * n_best : (j + 1) * n_best])
        print(f"  {min(i + batch_size, len(srcs))}/{len(srcs)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path/id (default: --model); NLLB fine-tunes share the "
                         "base tokenizer, so point here at the base model if a checkpoint's "
                         "tokenizer.json is from an incompatible tokenizers version")
    ap.add_argument("--n-best", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} n_best={args.n_best}")

    srcs = read_sources(args.split, args.limit)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    bos = tgt_bos_id(tok)

    cands = generate_nbest(srcs, model, tok, bos, device, args.n_best,
                           args.batch_size, args.max_new_tokens)

    os.makedirs(NBEST, exist_ok=True)
    suffix = f"_n{args.limit}" if args.limit else ""
    out = os.path.join(NBEST, f"{args.split}{suffix}_nbest.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for idx, candidates in enumerate(cands):
            f.write(json.dumps({"idx": idx, "candidates": candidates},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(cands)} sources x {args.n_best} candidates -> {out}")


if __name__ == "__main__":
    main()
