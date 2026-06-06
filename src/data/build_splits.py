"""Reconstruct article_id and build article-disjoint, seeded train/dev/test splits.

Strategy: Data10KFinal.csv stores articles as consecutive blocks (article order
preserved). Each t*.txt / FORMAT/*.txt source file is ONE article whose line 1 is
the English title. We collect those titles, walk the canonical corpus top-to-bottom,
and start a new article whenever a row's English equals a known title. Then we
GroupShuffleSplit by article_id (seed 42, 80/10/10) so no article spans two splits.
"""
import json
import os
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from normalize import looks_zawgyi, normalize_text, numeral_profile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")
CANON = os.path.join(RAW, "Data10KFinal.csv")
SRC_DIRS = [
    os.path.join(RAW, "DATASET-CREATION"),
    os.path.join(RAW, "DATASET-CREATION", "FORMAT"),
]
OUT = os.path.join(ROOT, "data", "processed")
SEED = 42
MYANMAR = re.compile(r"[က-႟]")


def norm(s: str) -> str:
    """Whitespace-collapsed, lowercased key for title matching."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def collect_titles() -> set:
    """Line 1 of each English per-article source file => a title key."""
    titles = set()
    for d in SRC_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith(".txt"):
                continue
            if fn.lower() in ("eng1.txt", "my1.txt"):  # aggregated, not per-article
                continue
            path = os.path.join(d, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    first = f.readline()
            except Exception:
                continue
            if not first.strip():
                continue
            # English title only: has Latin letters, little/no Myanmar
            if re.search(r"[A-Za-z]", first) and len(MYANMAR.findall(first)) <= 2:
                titles.add(norm(first))
    return titles


def main() -> None:
    df = pd.read_csv(CANON, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"my", "en"} <= set(df.columns), f"unexpected cols: {df.columns}"
    df = df[["en", "my"]].astype(str)
    # conservative normalization BEFORE dedup/split (reproducible, meaning-preserving)
    df["en"] = df["en"].map(normalize_text)
    df["my"] = df["my"].map(normalize_text)
    zawgyi_rows = int(df["my"].map(looks_zawgyi).sum())
    numeral_counts = df["my"].map(numeral_profile).value_counts().to_dict()

    # dedup exact pairs, keep first (preserves article order)
    before = len(df)
    df = df.drop_duplicates(subset=["en", "my"]).reset_index(drop=True)
    dropped = before - len(df)

    titles = collect_titles()
    en_keys = df["en"].map(norm)
    is_title = en_keys.isin(titles)

    # walk and assign article ids by title boundaries
    article_id = [-1] * len(df)
    cur = -1
    for i, t in enumerate(is_title.tolist()):
        if t:
            cur += 1
        article_id[i] = cur
    df["article_id"] = article_id

    n_titles_found = int(is_title.sum())
    leading_unassigned = int((df["article_id"] == -1).sum())

    # Rows before the first detected title have an unrecoverable article boundary.
    # Force them entirely into TRAIN (partial grouping is harmless there) so that
    # dev/test contain ONLY cleanly-bounded articles.
    residual = df[df["article_id"] == -1].reset_index(drop=True)
    clean = df[df["article_id"] >= 0].reset_index(drop=True)

    n_articles = clean["article_id"].nunique()
    sizes = clean.groupby("article_id").size()

    # article-disjoint split over cleanly-bounded articles: 80/10/10
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
    tr_idx, rest_idx = next(gss1.split(clean, groups=clean["article_id"]))
    rest = clean.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)
    dev_rel, test_rel = next(gss2.split(rest, groups=rest["article_id"]))
    train = pd.concat([clean.iloc[tr_idx], residual], ignore_index=True)
    dev = rest.iloc[dev_rel].reset_index(drop=True)
    test = rest.iloc[test_rel].reset_index(drop=True)

    # assert disjoint articles
    a_tr, a_dev, a_te = set(train.article_id), set(dev.article_id), set(test.article_id)
    assert not (a_tr & a_dev) and not (a_tr & a_te) and not (a_dev & a_te), "ARTICLE LEAKAGE!"

    os.makedirs(OUT, exist_ok=True)
    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for i, r in part.iterrows():
                f.write(json.dumps(
                    {"id": f"{name}-{i}", "article_id": int(r.article_id),
                     "en": r.en, "my": r.my}, ensure_ascii=False) + "\n")

    report = {
        "canonical_rows_before_dedup": before,
        "exact_dup_rows_dropped": dropped,
        "rows_after_dedup": len(df),
        "zawgyi_suspect_rows": zawgyi_rows,
        "myanmar_numeral_profile": numeral_counts,
        "title_boundaries_detected": n_titles_found,
        "distinct_articles": int(n_articles),
        "rows_before_first_title": leading_unassigned,
        "median_rows_per_article": int(sizes.median()),
        "min_rows_per_article": int(sizes.min()),
        "max_rows_per_article": int(sizes.max()),
        "split_rows": {"train": len(train), "dev": len(dev), "test": len(test)},
        "split_articles": {"train": len(a_tr), "dev": len(a_dev), "test": len(a_te)},
        "article_overlap": {"train_dev": len(a_tr & a_dev),
                            "train_test": len(a_tr & a_te),
                            "dev_test": len(a_dev & a_te)},
        "seed": SEED,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(os.path.join(ROOT, "data", "processed", "split_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
