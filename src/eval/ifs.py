"""Instruction Faithfulness Score (IFS) for en->my procedural MT.

IFS measures how faithfully a translation preserves the *procedural* content of a
source instruction, decomposed into four components:

  - quantity : numbers / quantities (Arabic + Burmese digits)
  - entity   : named/copyable entities (brands, acronyms, URLs, proper nouns)
  - step     : count of step / sub-step units (sentence-clause segmentation)
  - action   : imperative action verbs -- assessed via the human-validation
               protocol (no reliable open Burmese verb tagger), NOT automated here

The three automatic components form an automatic IFS proxy that is validated
against human IFS by correlation, and that doubles as a control signal for the
agentic-translation work (Track 2). IFS is source-anchored: it compares the
English source against the Burmese hypothesis, so the reference is not required.

Scope is deliberately honest per component: entity matching catches copied /
loanword entities (e.g. "TikTok") but not Burmese *transliterations* of names;
step faithfulness is near-degenerate on 1:1 sentence-aligned data and becomes
informative at article/agent scope. Those gaps are what the human protocol covers.

Usage:
  python src/eval/ifs.py --hyps experiments/results/nllb_finetuned_test_hyps.txt \
      --refs data/processed/test.jsonl --system nllb_finetuned
"""
import argparse
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "experiments", "results")

# Burmese digits U+1040..U+1049 map 1:1 onto Arabic 0..9.
_BURMESE_TO_ARABIC = {chr(0x1040 + i): str(i) for i in range(10)}
_NUMERAL_RE = re.compile(r"[0-9၀-၉]+(?:[.,][0-9၀-၉]+)?")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HANDLE_RE = re.compile(r"[@#]\w{2,}")
_CAMEL_RE = re.compile(r"\b[A-Za-z]+[A-Z][A-Za-z0-9]*\b")  # TikTok, YouTube, iPhone
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")                 # USB, PDF, HTML
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")
# Step/clause boundaries across both scripts: latin enders + Burmese ။ (U+104A),
# ၊ (U+104B), semicolons and newlines.
_STEP_SPLIT_RE = re.compile(r"[.!?;၊။\n]+")
_SENT_SPLIT_RE = re.compile(r"[.!?၊]\s+")

MIN_ENTITY_LEN = 2
IFS_WEIGHTS = {"quantity": 0.4, "entity": 0.4, "step": 0.2}


def read_lines(path: str) -> list:
    """Read hypothesis lines (one translation per line)."""
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def read_field(path: str, field: str) -> list:
    """Read one field from a jsonl file (e.g. 'en' sources)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[field] for line in f]


def extract_numerals(text: str) -> Counter:
    """Return a multiset of numeric tokens, Burmese digits normalised to Arabic."""
    out: Counter = Counter()
    for tok in _NUMERAL_RE.findall(text):
        out["".join(_BURMESE_TO_ARABIC.get(ch, ch) for ch in tok)] += 1
    return out


def quantity_faithfulness(src: str, hyp: str) -> float:
    """F1 of the numeral multiset shared between source and hypothesis (1.0 if none)."""
    src_nums, hyp_nums = extract_numerals(src), extract_numerals(hyp)
    if not src_nums and not hyp_nums:
        return 1.0
    shared = sum((src_nums & hyp_nums).values())
    precision = shared / sum(hyp_nums.values()) if hyp_nums else 0.0
    recall = shared / sum(src_nums.values()) if src_nums else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _proper_nouns(text: str) -> set:
    """Capitalised, non-sentence-initial words (a proper-noun heuristic)."""
    nouns = set()
    for sentence in _SENT_SPLIT_RE.split(text):
        words = _WORD_RE.findall(sentence)
        for word in words[1:]:  # skip sentence-initial capitalisation
            if word[0].isupper() and len(word) >= MIN_ENTITY_LEN:
                nouns.add(word)
    return nouns


def extract_entities(text: str) -> set:
    """Copyable/named entities a faithful translation should preserve verbatim."""
    ents = set(_URL_RE.findall(text)) | set(_HANDLE_RE.findall(text))
    ents |= set(_CAMEL_RE.findall(text)) | set(_ACRONYM_RE.findall(text))
    ents |= _proper_nouns(text)
    return {e for e in ents if len(e) >= MIN_ENTITY_LEN}


def entity_faithfulness(src: str, hyp: str) -> float:
    """Fraction of source entities present (case-insensitively) in the hypothesis."""
    entities = extract_entities(src)
    if not entities:
        return 1.0
    lowered = hyp.lower()
    present = sum(1 for entity in entities if entity.lower() in lowered)
    return present / len(entities)


def count_steps(text: str) -> int:
    """Count non-empty step/clause units (script-agnostic segmentation)."""
    return sum(1 for unit in _STEP_SPLIT_RE.split(text) if unit.strip())


def step_faithfulness(src: str, hyp: str) -> float:
    """Closeness of step/clause counts (min/max ratio; 1.0 if both empty)."""
    src_steps, hyp_steps = count_steps(src), count_steps(hyp)
    if max(src_steps, hyp_steps) == 0:
        return 1.0
    return min(src_steps, hyp_steps) / max(src_steps, hyp_steps)


def segment_ifs(src: str, hyp: str, weights: dict = IFS_WEIGHTS) -> dict:
    """Per-segment IFS: the automatic components plus their weighted overall."""
    components = {
        "quantity": quantity_faithfulness(src, hyp),
        "entity": entity_faithfulness(src, hyp),
        "step": step_faithfulness(src, hyp),
    }
    total = sum(weights[name] for name in components)
    overall = sum(weights[name] * value for name, value in components.items()) / total
    return {**components, "ifs": overall}


def corpus_ifs(srcs: list, hyps: list, system: str,
               weights: dict = IFS_WEIGHTS) -> dict:
    """Corpus-level IFS (component means + weighted overall), 0-100 scaled."""
    assert len(srcs) == len(hyps), f"{len(srcs)} srcs vs {len(hyps)} hyps"
    sums = Counter()
    for src, hyp in zip(srcs, hyps):
        for name, value in segment_ifs(src, hyp, weights).items():
            sums[name] += value
    n = len(hyps)
    scaled = {name: round(100 * total / n, 2) for name, total in sums.items()} if n else {}
    return {
        "system": system,
        "n": n,
        "ifs": scaled.get("ifs"),
        "ifs_quantity": scaled.get("quantity"),
        "ifs_entity": scaled.get("entity"),
        "ifs_step": scaled.get("step"),
        "ifs_weights": weights,
        "ifs_action_note": "action faithfulness assessed via human protocol, not automated",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--refs", required=True, help="jsonl with the source field")
    ap.add_argument("--src-field", default="en")
    ap.add_argument("--system", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N")
    ap.add_argument("--out", default=os.path.join(RESULTS, "ifs_results.json"))
    args = ap.parse_args()

    hyps = read_lines(args.hyps)
    srcs = read_field(args.refs, args.src_field)
    if args.limit:
        hyps, srcs = hyps[: args.limit], srcs[: args.limit]
    res = corpus_ifs(srcs, hyps, args.system)
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
