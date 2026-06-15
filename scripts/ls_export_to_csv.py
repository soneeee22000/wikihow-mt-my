"""Convert a Label Studio JSON export into the ratings CSV that correlate.py reads.

Label Studio (off-the-shelf annotation tool) is used to collect the IFS human
ratings via a shareable link. Export the project as JSON, then run this to produce
experiments/results/ratings_filled.csv (id, src_en, hyp_my, followability, adequacy,
and the binary component checks step_order/action_correct/entities_correct/
quantities_correct), which `src/eval/correlate.py --ratings ... --key ratings_key.csv`
analyses.

The component checks are Label Studio <Choices> (yes/no) widgets; the followability and
adequacy ratings are <Rating> widgets. We capture both widget kinds so the human "action"
IFS component (human-only) and the flagship's repair-targeting signal are not dropped on
export (see docs/flagship-rq4-protocol.md).

Handles multiple annotators: emits one row per (task, annotation) so inter-rater
agreement (Krippendorff's alpha) can be computed from the duplicates.

Usage:
  python scripts/ls_export_to_csv.py <labelstudio_export.json> [out.csv]
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "experiments", "results", "ratings_filled.csv")
COMPONENT_FIELDS = ["step_order", "action_correct", "entities_correct", "quantities_correct"]
FIELDS = ["id", "annotator", "src_en", "hyp_my", "followability", "adequacy"] + COMPONENT_FIELDS


def ratings_from_result(result: list) -> dict:
    """Pull {from_name: value} out of a Label Studio annotation 'result' list,
    covering both <Rating> (numeric) and <Choices> (first selected label) widgets."""
    out = {}
    for item in result:
        value = item.get("value", {})
        name = item.get("from_name")
        if "rating" in value:
            out[name] = value["rating"]
        elif value.get("choices"):
            out[name] = value["choices"][0]
    return out


def convert(export_path: str, out_path: str) -> int:
    """Write one CSV row per (task, annotation). Returns the row count."""
    with open(export_path, encoding="utf-8") as f:
        tasks = json.load(f)
    rows = []
    for task in tasks:
        data = task.get("data", {})
        for ann in (task.get("annotations") or []):
            vals = ratings_from_result(ann.get("result", []))
            if not vals:
                continue
            row = {
                "id": data.get("id", ""),
                "annotator": ann.get("completed_by", ""),
                "src_en": data.get("src_en", ""),
                "hyp_my": data.get("hyp_my", ""),
                "followability": vals.get("followability", ""),
                "adequacy": vals.get("adequacy", ""),
            }
            row.update({c: vals.get(c, "") for c in COMPONENT_FIELDS})
            rows.append(row)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/ls_export_to_csv.py <export.json> [out.csv]")
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    n = convert(sys.argv[1], out_path)
    print(f"wrote {n} rating rows -> {out_path}")


if __name__ == "__main__":
    main()
