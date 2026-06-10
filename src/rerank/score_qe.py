"""Reference-free QE scoring of the N-best candidates with COMET-Kiwi.

Scores every (source, candidate) pair with Unbabel/wmt22-cometkiwi-da (no
reference needed) and writes per-source QE lists to
experiments/results/nbest/<split>_qe.jsonl:
  {"idx": i, "qe": [q_0, ..., q_{N-1}]}

Runs in the isolated CPU comet venv (C:\\comet_venv) and is CHUNK-RESUMABLE: it
processes sources in chunks and skips idx already present in the output, so an
interrupted run (e.g. laptop close) resumes instead of restarting.

  C:\\comet_venv\\Scripts\\python.exe src/rerank/score_qe.py --split test
  ... --limit 8   # smoke (matches translate_nbest --limit 8 output name)

NOTE: Unbabel/wmt22-cometkiwi-da is a GATED model on the Hugging Face Hub. Accept
its license once at https://huggingface.co/Unbabel/wmt22-cometkiwi-da and export
HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before the first run.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
NBEST = os.path.join(ROOT, "experiments", "results", "nbest")
QE_MODEL = "Unbabel/wmt22-cometkiwi-da"
CHUNK_SOURCES = 50  # flush after this many sources (resume granularity)


def load_env_token() -> None:
    """Export HF_TOKEN from the repo .env so huggingface_hub can fetch the gated
    QE model. Dependency-free (this runs in the isolated COMET venv); a real env
    var already set always wins."""
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(key):
            return
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_HUB_TOKEN="):
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    os.environ["HF_TOKEN"] = val
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = val
                    return


def read_sources(split: str, limit: int) -> list:
    """Load en sources from data/processed/<split>.jsonl (first `limit` if >0)."""
    rows = []
    with open(os.path.join(PROC, f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return [r["en"] for r in rows]


def read_nbest(path: str) -> list:
    """Load [{idx, candidates:[...]}] rows from an nbest jsonl."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def done_idx(path: str) -> set:
    """idx values already scored in the output (for resume)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["idx"] for line in f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = full; else _n{limit} files")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--qe-model", default=QE_MODEL,
                    help="QE checkpoint id. Default (cometkiwi-da) is GATED; "
                         "'wmt20-comet-qe-da' is a non-gated legacy fallback")
    args = ap.parse_args()

    load_env_token()
    from comet import download_model, load_from_checkpoint
    import torch

    suffix = f"_n{args.limit}" if args.limit else ""
    nbest_path = os.path.join(NBEST, f"{args.split}{suffix}_nbest.jsonl")
    out_path = os.path.join(NBEST, f"{args.split}{suffix}_qe.jsonl")
    assert os.path.exists(nbest_path), f"missing {nbest_path}; run translate_nbest first"

    srcs = read_sources(args.split, args.limit)
    rows = read_nbest(nbest_path)
    assert len(rows) == len(srcs), f"{len(rows)} nbest rows vs {len(srcs)} sources"

    already = done_idx(out_path)
    todo = [r for r in rows if r["idx"] not in already]
    if not todo:
        print(f"{out_path} already complete ({len(already)} sources)")
        return
    print(f"QE: {len(todo)}/{len(rows)} sources to score (resuming past {len(already)})",
          flush=True)

    print(f"QE model: {args.qe_model}", flush=True)
    model = load_from_checkpoint(download_model(args.qe_model))
    gpus = 1 if torch.cuda.is_available() else 0

    with open(out_path, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), CHUNK_SOURCES):
            chunk = todo[start : start + CHUNK_SOURCES]
            data, spans = [], []
            for r in chunk:
                src = srcs[r["idx"]]
                spans.append((r["idx"], len(r["candidates"])))
                data.extend({"src": src, "mt": c} for c in r["candidates"])
            scores = model.predict(data, batch_size=args.batch_size, gpus=gpus)["scores"]
            pos = 0
            for idx, k in spans:
                out.write(json.dumps({"idx": idx, "qe": [round(float(s), 6)
                                      for s in scores[pos : pos + k]]}) + "\n")
                pos += k
            out.flush()
            print(f"  scored {min(start + CHUNK_SOURCES, len(todo))}/{len(todo)}", flush=True)
    print(f"wrote QE -> {out_path}")


if __name__ == "__main__":
    main()
