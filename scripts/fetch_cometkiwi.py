"""Robustly fetch the gated COMET-Kiwi QE model into the HF cache.

The HF API/tree endpoints have been intermittently resetting the connection on
this network (ConnectionResetError 10054), which leaves an incomplete cache (only
LICENSE+README, no checkpoints/model.ckpt). snapshot_download resumes partial LFS
blobs across attempts, so we retry a few times and then verify the .ckpt is present
and non-trivial in size. Run in the isolated COMET venv:

  C:\\comet_venv\\Scripts\\python.exe scripts/fetch_cometkiwi.py
"""
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "Unbabel/wmt22-cometkiwi-da"
MAX_ATTEMPTS = 8


def load_token() -> None:
    """Export HF_TOKEN from the repo .env (dependency-free)."""
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(key):
            return
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("HF_TOKEN=") or s.startswith("HUGGINGFACE_HUB_TOKEN="):
                val = s.partition("=")[2].strip().strip('"').strip("'")
                if val:
                    os.environ["HF_TOKEN"] = val
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = val
                    return


def main() -> None:
    load_token()
    from huggingface_hub import snapshot_download

    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[attempt {attempt}/{MAX_ATTEMPTS}] snapshot_download {REPO}", flush=True)
            path = snapshot_download(REPO, etag_timeout=60)
            ckpt = os.path.join(path, "checkpoints", "model.ckpt")
            if os.path.exists(ckpt) and os.path.getsize(ckpt) > 10_000_000:
                size_gb = os.path.getsize(ckpt) / 1e9
                print(f"OK: model.ckpt present ({size_gb:.2f} GB) -> {path}", flush=True)
                return
            print(f"incomplete: model.ckpt missing/small at {ckpt}; retrying", flush=True)
        except Exception as exc:  # transient network resets, etc.
            last_exc = exc
            print(f"[attempt {attempt}] {type(exc).__name__}: {str(exc)[:140]}", flush=True)
        time.sleep(min(10 * attempt, 60))
    raise SystemExit(f"FAILED after {MAX_ATTEMPTS} attempts; last error: {last_exc}")


if __name__ == "__main__":
    main()
