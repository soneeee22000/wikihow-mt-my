"""Push the fine-tuned NLLB-200-distilled-600M (en->my, WikiHow-MY) to the HF Hub.

Creates the model repo, writes a metric-driven model card (README.md with the
model-index YAML so the Hub renders the chrF++/BLEU/spBLEU leaderboard), and
uploads the local checkpoint folder (model.safetensors + tokenizer + configs).

Auth: run `hf auth login` once, or export HF_TOKEN. The CC-BY-NC-4.0 license is
inherited from NLLB-200 and is non-negotiable for this derivative (see the card).

Usage:
  python scripts/push_to_hub.py --repo-id <user>/nllb-600m-wikihow-en-my
  python scripts/push_to_hub.py --repo-id <user>/... --private --create-pr
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import EvalResult, HfApi, ModelCard, ModelCardData

CHECKPOINT_DIR = Path("checkpoints/nllb_finetuned_wikihow")
BASE_MODEL = "facebook/nllb-200-distilled-600M"
CARD_TEMPLATE = Path("paper/MODEL_CARD.md")

# From experiments/results/main_results.json (en->my, eng_Latn->mya_Mymr).
FLORES_CHRF = 33.5
FLORES_SPBLEU = 17.78
WIKIHOW_CHRF = 41.64
WIKIHOW_SPBLEU = 23.18


def _build_card(repo_id: str) -> ModelCard:
    """Build a ModelCard whose model-index drives the Hub metrics table.

    Loads the human-written body from paper/MODEL_CARD.md and overlays the
    EvalResult-generated model-index so the metric numbers live in one place.
    """
    model_name = repo_id.split("/")[-1]
    card_data = ModelCardData(
        language=["en", "my"],
        license="cc-by-nc-4.0",
        base_model=BASE_MODEL,
        library_name="transformers",
        pipeline_tag="translation",
        tags=["translation", "nllb", "burmese", "wikihow", "low-resource"],
        datasets=["facebook/flores"],
        metrics=["chrf", "bleu"],
        model_name=model_name,
        eval_results=[
            EvalResult(
                task_type="translation",
                task_name="Machine Translation (English to Burmese)",
                dataset_type="custom",
                dataset_name="WikiHow-MY (held-out test, article-disjoint)",
                dataset_split="test",
                metric_type="chrf",
                metric_name="chrF++",
                metric_value=WIKIHOW_CHRF,
            ),
            EvalResult(
                task_type="translation",
                task_name="Machine Translation (English to Burmese)",
                dataset_type="custom",
                dataset_name="WikiHow-MY (held-out test, article-disjoint)",
                dataset_split="test",
                metric_type="bleu",
                metric_name="spBLEU",
                metric_value=WIKIHOW_SPBLEU,
            ),
            EvalResult(
                task_type="translation",
                task_name="Machine Translation (English to Burmese)",
                dataset_type="facebook/flores",
                dataset_name="FLORES-200 (devtest, eng_Latn-mya_Mymr)",
                dataset_config="eng_Latn-mya_Mymr",
                dataset_split="devtest",
                metric_type="chrf",
                metric_name="chrF++",
                metric_value=FLORES_CHRF,
            ),
            EvalResult(
                task_type="translation",
                task_name="Machine Translation (English to Burmese)",
                dataset_type="facebook/flores",
                dataset_name="FLORES-200 (devtest, eng_Latn-mya_Mymr)",
                dataset_config="eng_Latn-mya_Mymr",
                dataset_split="devtest",
                metric_type="bleu",
                metric_name="spBLEU",
                metric_value=FLORES_SPBLEU,
            ),
        ],
    )
    body = CARD_TEMPLATE.read_text(encoding="utf-8")
    if body.lstrip().startswith("---"):
        body = body.split("---", 2)[-1].lstrip("\n")
    content = f"---\n{card_data.to_yaml()}\n---\n\n{body}"
    card = ModelCard(content)
    try:
        card.validate()  # online lint; best-effort (HF API can be flaky)
    except Exception as exc:
        print(f"[warn] online card validation skipped ({type(exc).__name__})", flush=True)
    return card


def _retry(label: str, fn, attempts: int = 30):
    """Run a network op with backoff; the HF API on this network resets connections
    intermittently (WinError 10054). All callers here are idempotent/resumable."""
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            print(f"[{label}] attempt {i}/{attempts}: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            time.sleep(min(8 * i, 60))
    raise last


def load_env_token() -> None:
    """Export HF_TOKEN from the repo .env so the upload authenticates without the CLI."""
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(key):
            return
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_HUB_TOKEN="):
            val = line.partition("=")[2].strip().strip('"').strip("'")
            if val:
                os.environ["HF_TOKEN"] = val
                os.environ["HUGGINGFACE_HUB_TOKEN"] = val
                return


def main() -> None:
    """Create the repo, push the card, then upload the checkpoint folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--create-pr", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_dir)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"No model.safetensors in {checkpoint}")

    load_env_token()
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    api = HfApi()

    _retry("create_repo", lambda: api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    ))

    card = _build_card(args.repo_id)
    _retry("push_card", lambda: card.push_to_hub(
        args.repo_id, repo_type="model", create_pr=args.create_pr))

    _retry("upload_folder", lambda: api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(checkpoint),
        commit_message="Add fine-tuned NLLB-200-distilled-600M (en->my, WikiHow-MY)",
        ignore_patterns=["training_args.bin", "checkpoint-*", "**/optimizer.pt", "*.log"],
        create_pr=args.create_pr,
    ))
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
