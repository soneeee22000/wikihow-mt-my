"""Kaggle GPU kernel entrypoint: N-best generation for the reranking booster.

Loads the fine-tuned NLLB checkpoint from the attached private dataset and decodes
an N-best list (beam search, num_return_sequences=N) for the dev and test splits,
eng_Latn->mya_Mymr. Mirrors src/infer/translate_nbest.py, but runs on the Kaggle
GPU because a beam-16 decode over ~1.75k sources is ~hours on CPU.

The fine-tune does not change the tokenizer, so the base NLLB tokenizer is loaded
from the Hub (internet on) to sidestep the checkpoint's tokenizer.json being from a
newer `tokenizers` version. Model weights load from the mounted checkpoint dir,
which is `model.safetensors` (safetensors -> no torch.load CVE gate, so Kaggle's
stock transformers can load it directly; no version pin needed on supported GPUs).

Environment strategy (learned the hard way):
- torchvision/torchaudio are uninstalled: NLLB (m2m_100) is text-only, and Kaggle's
  stock torchvision is often ABI-mismatched against its torch ("operator
  torchvision::nms does not exist"), which crashes the transformers import chain.
- The GPU arch is probed in a SUBPROCESS (so the parent never imports a torch we
  might replace). If stock torch already supports the arch (T4 sm_75), the stock,
  self-consistent torch+transformers are kept. Only an unsupported arch (Pascal
  P100 sm_60) triggers the proven torch==2.4.1 + transformers==4.46.3 downgrade.

Outputs one row per source to /kaggle/working/<split>_nbest.jsonl:
  {"idx": i, "candidates": [N strings]}   (candidate 0 = top beam)
pulled locally into experiments/results/nbest/ by scripts/kaggle_nbest.py.

Do not run locally; assumes the Kaggle filesystem + GPU.
"""
import glob
import json
import os
import subprocess
import sys

WORK = "/kaggle/working"
SMOKE = False  # flipped to True by `kaggle_nbest.py --smoke`
SMOKE_LIMIT = 8
SMOKE_N_BEST = 16  # == full N_BEST so the smoke exercises the real beam-16 GPU memory

N_BEST = 16
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
SRC_LANG = "eng_Latn"
TGT_LANG = "mya_Mymr"
BASE_TOKENIZER = "facebook/nllb-200-distilled-600M"
FALLBACK_TORCH = "2.4.1"          # cu121 wheel includes Pascal sm_60
FALLBACK_TRANSFORMERS = "4.46.3"  # proven to load NLLB on torch 2.4.1
SPLITS = ("dev", "test")
SEED = 42


def sh(cmd: str) -> None:
    """Run a shell command, streaming output, aborting the kernel on failure."""
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def probe_gpu() -> dict:
    """Probe torch/GPU in a SUBPROCESS so the parent never imports a torch we may
    replace. Returns {torch, cap, name, archs, supported} or {} if it failed."""
    code = (
        "import json, torch;"
        "cap = torch.cuda.get_device_capability(0);"
        "print('PROBE' + json.dumps({"
        "'torch': torch.__version__,"
        "'cap': 'sm_%d%d' % cap,"
        "'name': torch.cuda.get_device_name(0),"
        "'archs': torch.cuda.get_arch_list()}))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(res.stdout, res.stderr, flush=True)
    line = next((ln for ln in res.stdout.splitlines() if ln.startswith("PROBE")), None)
    if not line:
        return {}
    info = json.loads(line[len("PROBE"):])
    info["supported"] = any(info["cap"] in a for a in info["archs"])
    return info


def setup_env() -> None:
    """Make a self-consistent, text-only torch+transformers env before importing torch."""
    sh("pip uninstall -y -q torchvision torchaudio")
    info = probe_gpu()
    print(f"GPU probe: {info}", flush=True)
    if info and info.get("supported"):
        # Stock torch already supports this GPU (e.g. T4 sm_75). Keep Kaggle's
        # self-consistent torch+transformers; just make sure the helpers are present.
        sh("pip install -q sentencepiece accelerate protobuf")
        return
    # Unsupported arch (Pascal P100 sm_60) or probe failed: install the proven combo.
    print("stock torch lacks this GPU arch (or probe failed); installing proven combo",
          flush=True)
    sh(f"pip install -q torch=={FALLBACK_TORCH} "
       "--index-url https://download.pytorch.org/whl/cu121 "
       "--extra-index-url https://pypi.org/simple")
    sh(f"pip install -q 'transformers=={FALLBACK_TRANSFORMERS}' "
       "sentencepiece accelerate protobuf")


def find_checkpoint_dir() -> str:
    """Directory of the mounted fine-tuned checkpoint (parent of model.safetensors)."""
    hits = glob.glob("/kaggle/input/**/model.safetensors", recursive=True)
    assert hits, "model.safetensors not found in /kaggle/input — checkpoint not attached"
    return os.path.dirname(hits[0])


def find_split(split: str) -> str:
    """Path to the mounted <split>.jsonl source file."""
    hits = glob.glob(f"/kaggle/input/**/{split}.jsonl", recursive=True)
    assert hits, f"{split}.jsonl not found in /kaggle/input"
    return hits[0]


def read_sources(path: str, limit: int) -> list:
    """Load en sources from a split jsonl (first `limit` if >0)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return [r["en"] for r in rows]


def tgt_bos_id(tokenizer) -> int:
    """forced_bos_token_id for the target language, across transformers versions."""
    tid = tokenizer.convert_tokens_to_ids(TGT_LANG)
    if tid is not None and tid != tokenizer.unk_token_id:
        return tid
    return tokenizer.lang_code_to_id[TGT_LANG]


def generate_nbest(srcs, model, tok, bos, device, n_best, batch_size) -> list:
    """Return a list (per source) of n_best decoded candidates (top beam first)."""
    import torch
    out = []
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_NEW_TOKENS).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 num_beams=n_best, num_return_sequences=n_best,
                                 max_new_tokens=MAX_NEW_TOKENS)
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch)):
            out.append(decoded[j * n_best : (j + 1) * n_best])
        print(f"  {min(i + batch_size, len(srcs))}/{len(srcs)}", flush=True)
    return out


def run_split(split: str, ckpt_dir: str, n_best: int, limit: int) -> None:
    """Generate and write <split>_nbest.jsonl for one split."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(SEED)
    device = "cuda"
    src_path = find_split(split)
    srcs = read_sources(src_path, limit)
    print(f"[{split}] {len(srcs)} sources, n_best={n_best}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(device).eval()
    bos = tgt_bos_id(tok)

    cands = generate_nbest(srcs, model, tok, bos, device, n_best, BATCH_SIZE)
    out = os.path.join(WORK, f"{split}_nbest.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for idx, candidates in enumerate(cands):
            f.write(json.dumps({"idx": idx, "candidates": candidates},
                               ensure_ascii=False) + "\n")
    print(f"[{split}] wrote {len(cands)} x {n_best} -> {out}", flush=True)
    del model
    torch.cuda.empty_cache()


def main() -> None:
    setup_env()
    import torch
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
          f"| {torch.cuda.get_device_name(0)} | archs {torch.cuda.get_arch_list()}",
          flush=True)
    assert torch.cuda.is_available(), "no GPU on this kernel — enable the GPU accelerator"
    import transformers
    print(f"transformers {transformers.__version__}", flush=True)

    ckpt_dir = find_checkpoint_dir()
    print(f"checkpoint dir: {ckpt_dir}", flush=True)

    n_best = SMOKE_N_BEST if SMOKE else N_BEST
    limit = SMOKE_LIMIT if SMOKE else 0
    if SMOKE:
        print(f"=== SMOKE MODE: {limit} sources/split, n_best={n_best} ===", flush=True)

    for split in SPLITS:
        run_split(split, ckpt_dir, n_best, limit)

    print("\nPIPELINE DONE")


if __name__ == "__main__":
    main()
