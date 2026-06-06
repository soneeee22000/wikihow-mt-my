"""Batched NLLB inference, en->my. Runs on GPU (Colab) or CPU (slow, for smoke tests).

Writes hypotheses one-per-line to experiments/results/<system>_<split>_hyps.txt,
aligned with data/processed/<split>.jsonl.

Usage:
  python src/infer/translate.py --split test --system nllb_zeroshot --limit 20
  python src/infer/translate.py --split test --system nllb_finetuned --model <path>
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "experiments", "results")
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "mya_Mymr"
SEED = 42


def tgt_bos_id(tokenizer) -> int:
    """forced_bos_token_id for the target language, across transformers versions."""
    for fn in ("convert_tokens_to_ids",):
        tid = getattr(tokenizer, fn)(TGT_LANG)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    return tokenizer.lang_code_to_id[TGT_LANG]  # older API


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model}")

    rows = []
    with open(os.path.join(PROC, f"{args.split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    src = [r["en"] for r in rows]

    tok = AutoTokenizer.from_pretrained(args.model, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    bos = tgt_bos_id(tok)

    hyps = []
    for i in range(0, len(src), args.batch_size):
        batch = src[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_new_tokens).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 num_beams=args.num_beams,
                                 max_new_tokens=args.max_new_tokens)
        hyps.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  {min(i + args.batch_size, len(src))}/{len(src)}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    suffix = f"_n{args.limit}" if args.limit else ""
    out = os.path.join(RESULTS, f"{args.system}_{args.split}{suffix}_hyps.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps))
    print(f"wrote {len(hyps)} hyps -> {out}")


if __name__ == "__main__":
    main()
