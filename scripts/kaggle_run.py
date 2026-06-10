"""Drive the WikiHow-MY fine-tune on a Kaggle GPU, end to end, from this machine.

Stages a private Kaggle dataset (code.zip + the gitignored *.jsonl splits), pushes
a GPU script-kernel that runs scripts/kaggle/run_kernel.py, polls until it finishes,
and downloads the result artifacts back into experiments/results/ and paper/tables/.

The English WikiHow text never goes to GitHub; it rides in the *private* dataset only,
consistent with the rehydration release policy.

Prereq: ~/.kaggle/kaggle.json (Kaggle account -> Settings -> Create New API Token).

Usage:
  python scripts/kaggle_run.py              # full run: dataset + kernel + poll + pull
  python scripts/kaggle_run.py --status     # just poll the last kernel + pull if done
  python scripts/kaggle_run.py --no-wait    # push and exit (poll later with --status)
"""
import builtins
import sys

# The kaggle client writes downloaded logs with the platform default encoding;
# on Windows (cp1252) that crashes on non-ASCII (e.g. Burmese) kernel output.
# Force text-mode file I/O to utf-8 in-process (no re-exec, so stdout capture is
# preserved when this is launched in-session via `! python ...`).
_orig_open = builtins.open


def _utf8_open(*args, **kwargs):
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(*args, **kwargs)


builtins.open = _utf8_open
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import argparse
import json
import os
import re
import shutil
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "experiments", "results")
TABLES = os.path.join(ROOT, "paper", "tables")
KERNEL_SCRIPT = os.path.join(ROOT, "scripts", "kaggle", "run_kernel.py")

DATASET_SLUG = "wikihow-my-splits"
KERNEL_SLUG = "wikihow-my-ft"
SPLITS = ("train.jsonl", "dev.jsonl", "test.jsonl", "flores.jsonl")
POLL_SECONDS = 30
DONE_MARKER = "PIPELINE DONE"
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|CalledProcessError|"
    r"returned non-zero exit status|AssertionError|No matching distribution")


def api():
    """Authenticated Kaggle client (auths from ~/.kaggle/kaggle.json on import)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    client = KaggleApi()
    client.authenticate()
    return client


def username() -> str:
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), encoding="utf-8") as f:
        return json.load(f)["username"]


def zip_src(dest: str) -> None:
    """Zip src/ into dest as code.zip with 'src/...' arcnames for clean extraction."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(SRC):
            if "__pycache__" in folder:
                continue
            for name in files:
                if name.endswith(".pyc"):
                    continue
                full = os.path.join(folder, name)
                zf.write(full, os.path.relpath(full, ROOT))


def push_dataset(client, user: str) -> str:
    """Create or version the private dataset; return its full ref."""
    ref = f"{user}/{DATASET_SLUG}"
    stage = tempfile.mkdtemp(prefix="kgl_ds_")
    for split in SPLITS:
        shutil.copy(os.path.join(DATA, split), os.path.join(stage, split))
    zip_src(os.path.join(stage, "code.zip"))
    meta = {"title": DATASET_SLUG, "id": ref,
            "licenses": [{"name": "CC-BY-NC-SA-3.0"}]}
    with open(os.path.join(stage, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dataset] staging {ref} ({', '.join(SPLITS)} + code.zip)", flush=True)
    try:
        client.dataset_status(ref)
        existing = True
    except Exception:
        existing = False

    try:
        if existing:
            client.dataset_create_version(folder=stage, version_notes="update splits/code",
                                          convert_to_csv=False, dir_mode="skip",
                                          delete_old_versions=False)
            print("[dataset] new version pushed", flush=True)
        else:
            client.dataset_create_new(folder=stage, public=False,
                                      convert_to_csv=False, dir_mode="skip")
            print("[dataset] created new", flush=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    wait_dataset_ready(client, ref)
    # The new version needs extra server-side processing before it is mountable
    # by a kernel; a listable dataset is not yet an attachable one. Settle.
    print("[dataset] settling 45s before kernel push", flush=True)
    time.sleep(45)
    return ref


def wait_dataset_ready(client, ref: str) -> None:
    """Block until the dataset finishes processing so the kernel can attach it."""
    for _ in range(40):
        try:
            client.dataset_status(ref)  # raises until it exists/ready
            files = client.dataset_list_files(ref).files
            # Kaggle auto-extracts code.zip, so key off a top-level split instead.
            if any(getattr(f, "name", "").endswith("train.jsonl") for f in files):
                print("[dataset] ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(10)
    print("[dataset] proceeding (status check timed out; usually still fine)", flush=True)


def push_kernel(client, user: str, dataset_ref: str, smoke: bool = False) -> str:
    """Push the GPU script-kernel; return its full ref. If smoke, flip the kernel's
    SMOKE constant so it runs the tiny+fast end-to-end validation."""
    ref = f"{user}/{KERNEL_SLUG}"
    stage = tempfile.mkdtemp(prefix="kgl_k_")
    staged = os.path.join(stage, "run_kernel.py")
    shutil.copy(KERNEL_SCRIPT, staged)
    if smoke:
        with open(staged, encoding="utf-8") as f:
            text = f.read()
        text = text.replace("SMOKE = False", "SMOKE = True", 1)
        with open(staged, "w", encoding="utf-8") as f:
            f.write(text)
        print("[kernel] SMOKE mode enabled", flush=True)
    meta = {
        "id": ref,
        "title": KERNEL_SLUG,
        "code_file": "run_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(stage, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[kernel] pushing {ref} (GPU, internet, attached {dataset_ref})", flush=True)
    client.kernels_push(stage)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[kernel] running -> https://www.kaggle.com/code/{ref}", flush=True)
    return ref


def _parse_log(path: str) -> str:
    """Kaggle .log files are JSON arrays of {stream_name,time,data}; flatten to text."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    try:
        entries = json.loads(raw if raw.strip().startswith("[") else "[" + raw + "]")
        return "".join(e.get("data", "") for e in entries if isinstance(e, dict))
    except Exception:
        return raw


def fetch_output(client, ref: str):
    """Pull kernel output to a fresh temp dir. Returns (tmpdir, {name: path}, log_text).

    The caller owns tmpdir cleanup. Kaggle's status endpoint 500s in this SDK, so
    completion is inferred from the output artifacts + console log instead.
    """
    out = tempfile.mkdtemp(prefix="kgl_out_")
    try:
        client.kernels_output(ref, path=out)
    except Exception as exc:
        print(f"[kernel] output not ready ({str(exc)[:80]})", flush=True)
        return out, {}, ""
    files = {n: os.path.join(out, n) for n in os.listdir(out)
             if os.path.isfile(os.path.join(out, n))}
    log = next((_parse_log(p) for n, p in files.items() if n.endswith(".log")), "")
    return out, files, log


def baseline_log(client, ref: str) -> str:
    """Snapshot the current latest-version log so a fresh run can be told apart."""
    out, _, log = fetch_output(client, ref)
    shutil.rmtree(out, ignore_errors=True)
    return log


def poll(client, ref: str, baseline: str = ""):
    """Log/output-based completion detection. Returns (state, tmpdir, files)."""
    while True:
        out, files, log = fetch_output(client, ref)
        if "main_results.json" in files:
            print("[kernel] complete (results present)", flush=True)
            return "complete", out, files
        new_output = bool(log) and log != baseline
        if new_output and DONE_MARKER in log:
            print("[kernel] complete (pipeline done marker)", flush=True)
            return "complete", out, files
        if new_output and ERROR_RE.search(log):
            print("[kernel] error detected in log", flush=True)
            return "error", out, files
        shutil.rmtree(out, ignore_errors=True)
        print("[kernel] running... (no results yet)", flush=True)
        time.sleep(POLL_SECONDS)


def distribute(files: dict) -> None:
    """Copy pulled artifacts locally. JSON results are MERGED into existing files —
    the kernel only knows a subset of systems (NLLB + FLORES), so overwriting would
    drop the locally-scored API baselines (gemini/gtranslate) and IFS. .tex and .log
    are skipped (tables are regenerated locally from the merged JSON)."""
    os.makedirs(RESULTS, exist_ok=True)
    for name, src in files.items():
        if name.endswith(".tex") or name.endswith(".log"):
            continue
        dst = os.path.join(RESULTS, name)
        if name.endswith(".json") and os.path.exists(dst):
            local = json.load(open(dst, encoding="utf-8"))
            pulled = json.load(open(src, encoding="utf-8"))
            local.update(pulled)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(local, f, ensure_ascii=False, indent=2)
            print(f"[pull] merged {name} (+{len(pulled)} keys) -> {os.path.relpath(dst, ROOT)}", flush=True)
        else:
            shutil.copy(src, dst)
            print(f"[pull] {name} -> {os.path.relpath(dst, ROOT)}", flush=True)


def log_tail(files: dict, n: int = 2000) -> str:
    log = next((_parse_log(p) for nm, p in files.items() if nm.endswith(".log")), "")
    return log[-n:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="monitor existing kernel + pull")
    ap.add_argument("--pull", action="store_true", help="pull current output once, no wait")
    ap.add_argument("--no-wait", action="store_true", help="push then exit")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny+fast end-to-end validation run (catch bugs cheaply)")
    args = ap.parse_args()

    client = api()
    user = username()
    kernel_ref = f"{user}/{KERNEL_SLUG}"

    if args.pull:
        out, files, log = fetch_output(client, kernel_ref)
        distribute(files)
        if "main_results.json" not in files:
            print("[warn] no main_results.json yet; log tail:\n" + log[-2000:])
        shutil.rmtree(out, ignore_errors=True)
        return

    if args.status:
        state, out, files = poll(client, kernel_ref, baseline="")
    else:
        base = baseline_log(client, kernel_ref)
        dataset_ref = push_dataset(client, user)
        kernel_ref = push_kernel(client, user, dataset_ref, smoke=args.smoke)
        if args.no_wait:
            print("[done] pushed; monitor: python scripts/kaggle_run.py --status")
            return
        state, out, files = poll(client, kernel_ref, base)

    if state == "complete":
        if args.smoke:
            # Smoke output is tiny (n=16) — do NOT pull/merge it over the real results.
            print("[done] SMOKE complete (green). Results NOT distributed (smoke values).")
        else:
            distribute(files)
            print("[done] results pulled (merged).")
    else:
        print("[done] kernel ERRORED. Log tail:\n" + log_tail(files))
        print(f"Full log: https://www.kaggle.com/code/{kernel_ref}/log")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
