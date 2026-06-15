"""Contamination / leakage audit for the WikiHow-MY splits.

Two reviewer-facing questions this answers, both fully local:

1. Sentence-level leakage across the article-disjoint train/dev/test splits. Article-disjoint
   splitting does not by itself prevent identical *sentences* (boilerplate such as "Click OK.")
   from appearing in both train and test, which would let a model memorise test targets.
2. Independence of the FLORES+ control: our out-of-domain check is only valid if the WikiHow-MY
   test set does not overlap FLORES+.

Run: python -m src.data.check_contamination
Writes experiments/results/contamination.json and prints a summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

PROCESSED = Path("data/processed")
OUT_PATH = Path("experiments/results/contamination.json")
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting differences do not hide a match."""
    return _WS.sub(" ", text.strip().lower())


def _load(name: str) -> list[dict]:
    """Read one processed split (.jsonl) into a list of records."""
    path = PROCESSED / name
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _src_set(rows: Iterable[dict], *, normalize: bool) -> set[str]:
    """Set of source-English sentences, optionally normalised."""
    return {(_norm(r["en"]) if normalize else r["en"]) for r in rows}


def _pair_set(rows: Iterable[dict], *, normalize: bool) -> set[tuple[str, str]]:
    """Set of (source, target) pairs, optionally normalised on both sides."""
    if normalize:
        return {(_norm(r["en"]), _norm(r["my"])) for r in rows}
    return {(r["en"], r["my"]) for r in rows}


def _overlap(a: list[dict], b: list[dict]) -> dict[str, int]:
    """Counts of shared sources and shared pairs between two splits (raw and normalised)."""
    return {
        "src_exact": len(_src_set(a, normalize=False) & _src_set(b, normalize=False)),
        "src_norm": len(_src_set(a, normalize=True) & _src_set(b, normalize=True)),
        "pair_exact": len(_pair_set(a, normalize=False) & _pair_set(b, normalize=False)),
        "pair_norm": len(_pair_set(a, normalize=True) & _pair_set(b, normalize=True)),
    }


def _article_overlap(a: list[dict], b: list[dict]) -> int:
    """Number of article_ids shared between two splits (expected 0 for the disjoint split)."""
    ids_a = {r.get("article_id") for r in a if "article_id" in r}
    ids_b = {r.get("article_id") for r in b if "article_id" in r}
    return len(ids_a & ids_b)


def main() -> None:
    """Run all checks and persist the report."""
    train, dev, test = _load("train.jsonl"), _load("dev.jsonl"), _load("test.jsonl")
    flores = _load("flores.jsonl")

    report = {
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test), "flores": len(flores)},
        "article_overlap": {
            "test_vs_train": _article_overlap(test, train),
            "test_vs_dev": _article_overlap(test, dev),
            "dev_vs_train": _article_overlap(dev, train),
        },
        "test_vs_train": _overlap(test, train),
        "test_vs_dev": _overlap(test, dev),
        "test_vs_flores": _overlap(test, flores),
        "test_internal_dup_src": len(test) - len(_src_set(test, normalize=True)),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
