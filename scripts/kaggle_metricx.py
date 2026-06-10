"""Drive MetricX-24 scoring of the benchmark on a Kaggle GPU, end to end.

Stages a small private dataset of per-system MetricX input JSONLs (built by
src/eval/build_metricx_inputs.py), pushes a GPU kernel that runs
scripts/kaggle/run_metricx.py, polls until done, and merges the MetricX-24 scores
into experiments/results/main_results.json (under each system's "metricx24" key).

Reuses the low-level Kaggle helpers from kaggle_run.py (auth, dataset-ready wait,
log/output-based completion poll) so the proven workarounds aren't duplicated.

Usage:
  python scripts/kaggle_metricx.py --smoke   # tiny: large model, 8 rows, 2 systems
  python scripts/kaggle_metricx.py           # full: xl model, all systems
  python scripts/kaggle_metricx.py --status  # poll the last kernel + merge if done
"""
import argparse
import json
import os
import shutil
import tempfile

import kaggle_run as kr  # noqa: E402  (kaggle_run applies a utf-8 open() monkeypatch on import)

ROOT = kr.ROOT
RESULTS = kr.RESULTS
INPUTS = os.path.join(RESULTS, "metricx_inputs")
KERNEL_SCRIPT = os.path.join(ROOT, "scripts", "kaggle", "run_metricx.py")
DATASET_SLUG = "wikihow-my-metricx"
KERNEL_SLUG = "wikihow-metricx-run"  # must differ from the dataset slug (shared URL namespace)
METRIC_KEYS = ("metricx24", "metricx24_model")


def push_dataset(client, user: str) -> str:
    """Create/version the private dataset holding the MetricX input JSONLs."""
    ref = f"{user}/{DATASET_SLUG}"
    stage = tempfile.mkdtemp(prefix="kgl_mx_ds_")
    files = [f for f in os.listdir(INPUTS) if f.endswith(".jsonl")]
    assert files, f"no input jsonls in {INPUTS}; run src/eval/build_metricx_inputs.py first"
    # Stage at the dataset root (not a subfolder): dir_mode="skip" drops subfolders.
    for name in files:
        shutil.copy(os.path.join(INPUTS, name), os.path.join(stage, name))
    meta = {"title": DATASET_SLUG, "id": ref, "licenses": [{"name": "CC-BY-NC-SA-3.0"}]}
    with open(os.path.join(stage, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dataset] staging {ref} ({len(files)} input jsonls)", flush=True)
    try:
        client.dataset_status(ref)
        existing = True
    except Exception:
        existing = False
    try:
        if existing:
            client.dataset_create_version(folder=stage, version_notes="update metricx inputs",
                                          convert_to_csv=False, dir_mode="skip",
                                          delete_old_versions=False)
            print("[dataset] new version pushed", flush=True)
        else:
            client.dataset_create_new(folder=stage, public=False,
                                      convert_to_csv=False, dir_mode="skip")
            print("[dataset] created new", flush=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    _wait_ready(client, ref)
    print("[dataset] settling 45s before kernel push", flush=True)
    import time
    time.sleep(45)
    return ref


def _wait_ready(client, ref: str) -> None:
    """Block until a *.jsonl input is listable on the dataset (mountable)."""
    import time
    for _ in range(40):
        try:
            client.dataset_status(ref)
            files = client.dataset_list_files(ref).files
            if any(getattr(f, "name", "").endswith(".jsonl") for f in files):
                print("[dataset] ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(10)
    print("[dataset] proceeding (status check timed out; usually still fine)", flush=True)


def push_kernel(client, user: str, dataset_ref: str, smoke: bool) -> str:
    """Push the GPU kernel; flip SMOKE in the staged script when requested."""
    ref = f"{user}/{KERNEL_SLUG}"
    stage = tempfile.mkdtemp(prefix="kgl_mx_k_")
    staged = os.path.join(stage, "run_metricx.py")
    shutil.copy(KERNEL_SCRIPT, staged)
    if smoke:
        with open(staged, encoding="utf-8") as f:
            text = f.read()
        text = text.replace("SMOKE = False", "SMOKE = True", 1)
        with open(staged, "w", encoding="utf-8") as f:
            f.write(text)
        print("[kernel] SMOKE mode enabled", flush=True)
    meta = {
        "id": ref, "title": KERNEL_SLUG, "code_file": "run_metricx.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [dataset_ref], "competition_sources": [], "kernel_sources": [],
    }
    with open(os.path.join(stage, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[kernel] pushing {ref} (GPU, internet, attached {dataset_ref})", flush=True)
    client.kernels_push(stage)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[kernel] running -> https://www.kaggle.com/code/{ref}", flush=True)
    return ref


def merge(files: dict) -> None:
    """Merge metricx_results.json into main_results.json (per-system 'metricx24')."""
    src = files.get("metricx_results.json")
    if not src:
        print("[merge] no metricx_results.json in output", flush=True)
        return
    pulled = json.load(open(src, encoding="utf-8"))
    main_path = os.path.join(RESULTS, "main_results.json")
    main = json.load(open(main_path, encoding="utf-8"))
    for system, vals in pulled.items():
        main.setdefault(system, {})
        for k in METRIC_KEYS:
            if k in vals:
                main[system][k] = vals[k]
    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main, f, ensure_ascii=False, indent=2)
    print(f"[merge] MetricX-24 -> {main_path} ({len(pulled)} systems)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    client = kr.api()
    user = kr.username()
    kernel_ref = f"{user}/{KERNEL_SLUG}"

    if args.status:
        state, out, files = kr.poll(client, kernel_ref, baseline="")
    else:
        base = kr.baseline_log(client, kernel_ref)
        dataset_ref = push_dataset(client, user)
        kernel_ref = push_kernel(client, user, dataset_ref, smoke=args.smoke)
        if args.no_wait:
            print("[done] pushed; monitor: python scripts/kaggle_metricx.py --status")
            return
        state, out, files = kr.poll(client, kernel_ref, base)

    if state == "complete":
        if args.smoke:
            print("[done] SMOKE complete (green). Not merged (smoke values).")
            print("metricx_results.json:",
                  json.load(open(files["metricx_results.json"], encoding="utf-8"))
                  if "metricx_results.json" in files else "(missing)")
        else:
            merge(files)
            print("[done] MetricX-24 merged.")
    else:
        print("[done] kernel ERRORED. Log tail:\n" + kr.log_tail(files))
        print(f"Full log: https://www.kaggle.com/code/{kernel_ref}/log")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
