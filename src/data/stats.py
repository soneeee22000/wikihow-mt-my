"""Dataset statistics + a paper-ready figure, computed from data/processed/*.jsonl.

Myanmar is unsegmented, so we report MY length in characters (excluding spaces)
alongside whitespace-token counts; English length is whitespace tokens.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")
FIGDIR = os.path.join(ROOT, "paper", "figures")
NONSPACE = re.compile(r"\S")


def load(split):
    rows = []
    with open(os.path.join(PROC, f"{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    splits = {s: load(s) for s in ("train", "dev", "test")}
    allrows = [r for rs in splits.values() for r in rs]

    en_tok = [len(r["en"].split()) for r in allrows]
    my_chars = [len(NONSPACE.findall(r["my"])) for r in allrows]
    my_tok = [len(r["my"].split()) for r in allrows]

    # sentences per article (clean articles only, id >= 0)
    from collections import Counter
    art = Counter(r["article_id"] for r in allrows if r["article_id"] >= 0)
    spa = sorted(art.values())

    def summ(xs):
        xs = sorted(xs)
        n = len(xs)
        return {"n": n, "min": xs[0], "max": xs[-1],
                "mean": round(sum(xs) / n, 1), "median": xs[n // 2],
                "p95": xs[int(0.95 * n)]}

    stats = {
        "total_pairs": len(allrows),
        "split_sizes": {s: len(rs) for s, rs in splits.items()},
        "distinct_articles_clean": len(art),
        "sentences_per_article": summ(spa),
        "english_tokens": summ(en_tok),
        "myanmar_chars": summ(my_chars),
        "myanmar_whitespace_tokens": summ(my_tok),
    }
    os.makedirs(PROC, exist_ok=True)
    with open(os.path.join(PROC, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # figure
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].hist(en_tok, bins=40, color="#2b6cb0")
    ax[0, 0].set_title("English length (whitespace tokens)")
    ax[0, 0].set_xlabel("tokens"); ax[0, 0].set_ylabel("pairs")
    ax[0, 1].hist(my_chars, bins=40, color="#c05621")
    ax[0, 1].set_title("Myanmar length (characters)")
    ax[0, 1].set_xlabel("characters"); ax[0, 1].set_ylabel("pairs")
    ax[1, 0].hist(spa, bins=30, color="#2f855a")
    ax[1, 0].set_title("Sentences per article")
    ax[1, 0].set_xlabel("sentences"); ax[1, 0].set_ylabel("articles")
    names = list(splits.keys())
    ax[1, 1].bar(names, [len(splits[s]) for s in names],
                 color=["#2b6cb0", "#c05621", "#2f855a"])
    ax[1, 1].set_title("Split sizes (article-disjoint)")
    ax[1, 1].set_ylabel("pairs")
    fig.suptitle("WikiHow-MY English–Myanmar instructional MT corpus", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "dataset_stats.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
