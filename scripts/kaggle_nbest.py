"""Drive N-best generation for the reranking booster on a Kaggle GPU, end to end.

Stages a private dataset holding the fine-tuned NLLB checkpoint (~2.4 GB) plus the
dev/test source splits, pushes a GPU kernel that runs scripts/kaggle/run_nbest.py,
polls until done, and pulls the per-split N-best JSONLs into
experiments/results/nbest/ (gitignored: Burmese candidates, no English source).

Reuses the low-level Kaggle helpers from kaggle_run.py (auth, dataset-ready wait,
log/output-based completion poll). The big checkpoint is hard-linked into the upload
stage when possible to avoid a 2.4 GB copy.

Usage:
  python scripts/kaggle_nbest.py --smoke    # tiny: 8 sources/split, n_best=4 (validate path)
  python scripts/kaggle_nbest.py            # full: dev+test, n_best=16
  python scripts/kaggle_nbest.py --status   # poll the last kernel + pull if done
  python scripts/kaggle_nbest.py --no-wait  # push then exit
"""
import argparse
import json
import os
import shutil
import tempfile

import kaggle_run as kr  # noqa: E402  (applies the utf-8 open() monkeypatch on import)

ROOT = kr.ROOT
DATA = kr.DATA
RESULTS = kr.RESULTS
NBEST = os.path.join(RESULTS, "nbest")
KERNEL_SCRIPT = os.path.join(ROOT, "scripts", "kaggle", "run_nbest.py")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "nllb_finetuned_wikihow")
DATASET_SLUG = "wikihow-my-nllb-ckpt"
KERNEL_SLUG = "wikihow-nbest-run"  # must differ from the dataset slug (shared namespace)
SPLITS = ("dev.jsonl", "test.jsonl")
BIG_FILE = "model.safetensors"  # hard-link rather than copy


def _stage_file(src: str, dst: str) -> None:
    """Hard-link src->dst (cheap for the 2.4 GB weights); fall back to copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def push_dataset(client, user: str) -> str:
    """Create/version the private dataset: checkpoint files + dev/test splits."""
    ref = f"{user}/{DATASET_SLUG}"
    assert os.path.isdir(CKPT_DIR), f"missing checkpoint {CKPT_DIR}"
    stage = tempfile.mkdtemp(prefix="kgl_nb_ds_")
    ckpt_files = [f for f in os.listdir(CKPT_DIR)
                  if os.path.isfile(os.path.join(CKPT_DIR, f))]
    for name in ckpt_files:
        _stage_file(os.path.join(CKPT_DIR, name), os.path.join(stage, name))
    for split in SPLITS:
        shutil.copy(os.path.join(DATA, split), os.path.join(stage, split))
    meta = {"title": DATASET_SLUG, "id": ref, "licenses": [{"name": "CC-BY-NC-SA-3.0"}]}
    with open(os.path.join(stage, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dataset] staging {ref} ({len(ckpt_files)} ckpt files + {', '.join(SPLITS)})",
          flush=True)
    try:
        client.dataset_status(ref)
        existing = True
    except Exception:
        existing = False
    try:
        if existing:
            client.dataset_create_version(folder=stage, version_notes="update ckpt/splits",
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
    """Block until the checkpoint weights are listable on the dataset (mountable)."""
    import time
    for _ in range(60):  # ckpt upload + processing is slower than a few jsonls
        try:
            client.dataset_status(ref)
            files = client.dataset_list_files(ref).files
            if any(getattr(f, "name", "") == BIG_FILE for f in files):
                print("[dataset] ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(10)
    print("[dataset] proceeding (status check timed out; usually still fine)", flush=True)


def push_kernel(client, user: str, dataset_ref: str, smoke: bool) -> str:
    """Push the GPU kernel; flip SMOKE in the staged script when requested."""
    ref = f"{user}/{KERNEL_SLUG}"
    stage = tempfile.mkdtemp(prefix="kgl_nb_k_")
    staged = os.path.join(stage, "run_nbest.py")
    shutil.copy(KERNEL_SCRIPT, staged)
    if smoke:
        with open(staged, encoding="utf-8") as f:
            text = f.read()
        text = text.replace("SMOKE = False", "SMOKE = True", 1)
        with open(staged, "w", encoding="utf-8") as f:
            f.write(text)
        print("[kernel] SMOKE mode enabled", flush=True)
    meta = {
        "id": ref, "title": KERNEL_SLUG, "code_file": "run_nbest.py",
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


def pull(files: dict) -> None:
    """Copy <split>_nbest.jsonl artifacts into experiments/results/nbest/."""
    os.makedirs(NBEST, exist_ok=True)
    pulled = 0
    for name, src in files.items():
        if name.endswith("_nbest.jsonl"):
            shutil.copy(src, os.path.join(NBEST, name))
            n = sum(1 for _ in open(src, encoding="utf-8"))
            print(f"[pull] {name} ({n} sources) -> experiments/results/nbest/{name}",
                  flush=True)
            pulled += 1
    if not pulled:
        print("[pull] no *_nbest.jsonl in output", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--kernel-only", action="store_true",
                    help="reuse the existing dataset; push+poll the kernel only "
                         "(skips the 2.4 GB checkpoint re-upload)")
    args = ap.parse_args()

    client = kr.api()
    user = kr.username()
    kernel_ref = f"{user}/{KERNEL_SLUG}"

    if args.status:
        state, out, files = kr.poll(client, kernel_ref, baseline="")
    else:
        base = kr.baseline_log(client, kernel_ref)
        dataset_ref = (f"{user}/{DATASET_SLUG}" if args.kernel_only
                       else push_dataset(client, user))
        kernel_ref = push_kernel(client, user, dataset_ref, smoke=args.smoke)
        if args.no_wait:
            print("[done] pushed; monitor: python scripts/kaggle_nbest.py --status")
            return
        state, out, files = kr.poll(client, kernel_ref, base)

    if state == "complete":
        if args.smoke:
            print("[done] SMOKE complete (green). Not pulled (smoke values).")
            for n in files:
                if n.endswith("_nbest.jsonl"):
                    rows = [json.loads(x) for x in open(files[n], encoding="utf-8")]
                    print(f"  {n}: {len(rows)} sources, "
                          f"{len(rows[0]['candidates']) if rows else 0} candidates each")
        else:
            pull(files)
            print("[done] N-best pulled. Next: QE -> rerank tune -> eval.")
    else:
        print("[done] kernel ERRORED. Log tail:\n" + kr.log_tail(files))
        print(f"Full log: https://www.kaggle.com/code/{kernel_ref}/log")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
