"""Gemini 2.5 few-shot and Google Translate baselines for en->my WikiHow MT.

API-based and GPU-free. Writes hypotheses one-per-line to
experiments/results/<system>_<split>_hyps.txt, aligned with
data/processed/<split>.jsonl, so src/eval/automatic.py and src/eval/ifs.py score
them unchanged. Supports resume (skips already-written lines) and sanitises
newlines so each hypothesis stays on one physical line (alignment is positional).

Keys (environment variables):
  GEMINI_API_KEY            -- for --system gemini (Google AI Studio key)
  GOOGLE_TRANSLATE_API_KEY  -- for --system gtranslate (Cloud Translation API v2)

Usage:
  python src/infer/llm_baselines.py --system gemini --split test
  python src/infer/llm_baselines.py --system gtranslate --split test
"""
import argparse
import json
import os
import random
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "experiments", "results")

GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GTRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
FEWSHOT_K = 5
SEED = 42
REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
GTRANSLATE_BATCH = 50
GEMINI_PROGRESS_EVERY = 25


def load_dotenv() -> None:
    """Load KEY=VALUE lines from a gitignored project-root .env into os.environ
    (so API keys never need to be typed on the command line / pasted into chat)."""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_rows(split: str) -> list:
    """Load split rows ({en, my, ...}) from data/processed/<split>.jsonl."""
    with open(os.path.join(PROC, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def few_shot_examples(k: int = FEWSHOT_K) -> list:
    """Deterministically sample k en->my demonstration pairs from the train split."""
    train = read_rows("train")
    idx = random.Random(SEED).sample(range(len(train)), k)
    return [(train[i]["en"], train[i]["my"]) for i in idx]


def _sanitize(text: str) -> str:
    """Collapse all whitespace so one hypothesis stays on one line (positional alignment)."""
    return " ".join(text.split()).strip()


def _post(url: str, params: dict, payload: dict) -> dict:
    """POST JSON with exponential-backoff retry; raise on persistent failure."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, params=params, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            last = f"{resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after {MAX_RETRIES} tries: {last}")


def gemini_prompt(src: str, examples: list) -> str:
    """Few-shot instructional en->my translation prompt."""
    shots = "\n".join(f"English: {en}\nBurmese: {my}" for en, my in examples)
    return ("Translate the following English wikiHow instruction sentence into natural, "
            "fluent Burmese (Myanmar). Preserve every step, action, named entity, number "
            "and unit. Output only the Burmese translation.\n\n"
            f"{shots}\n\nEnglish: {src}\nBurmese:")


def gemini_translate(src: str, examples: list, model: str, key: str) -> str:
    """Translate one sentence via the Gemini generateContent REST API (deterministic)."""
    payload = {"contents": [{"parts": [{"text": gemini_prompt(src, examples)}]}],
               "generationConfig": {"temperature": 0.0}}
    data = _post(GEMINI_URL.format(model=model), {"key": key}, payload)
    cands = data.get("candidates")
    if not cands:  # blocked / empty -> empty hyp keeps alignment, scores honestly
        return ""
    parts = cands[0].get("content", {}).get("parts", [])
    return _sanitize("".join(p.get("text", "") for p in parts))


def gtranslate_batch(srcs: list, key: str) -> list:
    """Translate a batch of sentences via Cloud Translation API v2."""
    payload = {"q": srcs, "source": "en", "target": "my", "format": "text"}
    data = _post(GTRANSLATE_URL, {"key": key}, payload)
    return [_sanitize(t["translatedText"]) for t in data["data"]["translations"]]


def out_path(system: str, split: str) -> str:
    """experiments/results/<system>_<split>_hyps.txt (the harness's expected name)."""
    return os.path.join(RESULTS, f"{system}_{split}_hyps.txt")


def already_done(path: str) -> int:
    """Count hyps already written, so an interrupted run can resume."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def run_gemini(srcs: list, start: int, path: str, model: str, sleep: float) -> None:
    """Write Gemini hyps from index `start` onward, flushing each line for resume."""
    key = os.environ["GEMINI_API_KEY"]
    examples = few_shot_examples()
    with open(path, "a", encoding="utf-8") as out:
        for i in range(start, len(srcs)):
            out.write(gemini_translate(srcs[i], examples, model, key) + "\n")
            out.flush()
            if (i + 1) % GEMINI_PROGRESS_EVERY == 0:
                print(f"  {i + 1}/{len(srcs)}", flush=True)
            if sleep:
                time.sleep(sleep)


def run_gtranslate(srcs: list, start: int, path: str) -> None:
    """Write Google Translate hyps from index `start` onward in batches."""
    key = os.environ["GOOGLE_TRANSLATE_API_KEY"]
    with open(path, "a", encoding="utf-8") as out:
        for i in range(start, len(srcs), GTRANSLATE_BATCH):
            for hyp in gtranslate_batch(srcs[i : i + GTRANSLATE_BATCH], key):
                out.write(hyp + "\n")
            out.flush()
            print(f"  {min(i + GTRANSLATE_BATCH, len(srcs))}/{len(srcs)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["gemini", "gtranslate"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=GEMINI_MODEL, help="Gemini model id")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between Gemini calls")
    args = ap.parse_args()

    load_dotenv()
    rows = read_rows(args.split)
    if args.limit:
        rows = rows[: args.limit]
    srcs = [r["en"] for r in rows]

    os.makedirs(RESULTS, exist_ok=True)
    path = out_path(args.system, args.split)
    start = already_done(path)
    if start >= len(srcs):
        print(f"{path} already complete ({start} hyps)")
        return
    print(f"{args.system}: {len(srcs)} sources, resuming from {start}")

    if args.system == "gemini":
        run_gemini(srcs, start, path, args.model, args.sleep)
    else:
        run_gtranslate(srcs, start, path)
    print(f"wrote -> {path}")


if __name__ == "__main__":
    main()
