"""Kaggle GPU kernel entrypoint for the WikiHow-MY NLLB fine-tune pipeline.

Runs headless on a Kaggle GPU runtime (pushed via scripts/kaggle_run.py). It
reconstructs the project from the attached private dataset (code.zip + the three
*.jsonl splits, since the English text is gitignored and never on GitHub), then
runs: fine-tune -> inference (zero-shot + fine-tuned) -> score (chrF++/spBLEU/
BLEU/COMET) -> LaTeX table. Result artifacts are copied into /kaggle/working so
they come back in the kernel output.

Do not run this locally; it assumes the Kaggle filesystem (/kaggle/input,
/kaggle/working) and a GPU.
"""
import glob
import os
import shutil
import subprocess
import time
import zipfile

REPO = "/tmp/repo"
CKPT = "/tmp/ckpt"          # kept off /kaggle/working so the 2 GB checkpoint isn't downloaded
WORK = "/kaggle/working"
SMOKE = False  # flipped to True by `kaggle_run.py --smoke`: tiny+fast end-to-end check
SMOKE_LIMIT = 16
# COMET (unbabel-comet) is intentionally NOT installed here: it pins an ancient
# pytorch-lightning==1.6.4 whose metadata is rejected by modern pip on Kaggle's
# Python 3.12, and it would gate the whole GPU run. COMET is computed separately
# from the pulled hypotheses (same model/data). Kaggle does train + infer + chrF.
DEPS = ("transformers", "datasets", "sacrebleu",
        "sentencepiece", "pyyaml", "accelerate")
ARTIFACTS = (
    "experiments/results/main_results.json",
    "experiments/results/nllb_zeroshot_test_hyps.txt",
    "experiments/results/nllb_finetuned_test_hyps.txt",
    "experiments/results/nllb_zeroshot_flores_hyps.txt",
    "experiments/results/nllb_finetuned_flores_hyps.txt",
    "paper/tables/main_results.tex",
)


def sh(cmd: str) -> None:
    """Run a shell command, streaming output, aborting the kernel on failure."""
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def _discover():
    """Find the code (zip or extracted) and jsonl splits anywhere under /kaggle/input."""
    code_zip = next(iter(glob.glob("/kaggle/input/**/code.zip", recursive=True)), None)
    marker = next(iter(glob.glob(
        "/kaggle/input/**/src/train/finetune_nllb.py", recursive=True)), None)
    splits = glob.glob("/kaggle/input/**/*.jsonl", recursive=True)
    return code_zip, marker, splits


def stage_project() -> None:
    """Rebuild /tmp/repo from the attached dataset.

    Kaggle's input mount can be racy/partially-populated right at kernel start and
    may auto-extract zips, so the src/ tree may arrive as code.zip or already
    extracted under .../code/src/. Retry discovery until the mount settles, then
    handle whichever layout is present. Always dumps the input tree for ground truth.
    """
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    os.makedirs(REPO)

    code_zip = marker = None
    splits = []
    for _ in range(18):  # up to ~90s for the input mount to settle
        code_zip, marker, splits = _discover()
        if (code_zip or marker) and splits:
            break
        time.sleep(5)

    tree = [os.path.join(r, fn) for r, _, fs in os.walk("/kaggle/input") for fn in fs]
    print(f"INPUT TREE ({len(tree)} files):", tree[:80], flush=True)

    if marker:
        src_root = os.path.dirname(os.path.dirname(os.path.dirname(marker)))
        shutil.copytree(os.path.join(src_root, "src"), os.path.join(REPO, "src"))
    elif code_zip:
        with zipfile.ZipFile(code_zip) as zf:
            zf.extractall(REPO)

    assert os.path.exists(os.path.join(REPO, "src", "train", "finetune_nllb.py")), \
        "staging failed: src/ not found in /kaggle/input (see INPUT TREE above)"
    assert splits, "staging failed: no *.jsonl in /kaggle/input (see INPUT TREE above)"

    proc = os.path.join(REPO, "data", "processed")
    os.makedirs(proc, exist_ok=True)
    seen = {}
    for path in splits:  # de-dupe by basename
        seen.setdefault(os.path.basename(path), path)
    for name, path in seen.items():
        shutil.copy(path, os.path.join(proc, name))
    print("staged src:", sorted(os.listdir(os.path.join(REPO, "src"))),
          "| splits:", sorted(seen), flush=True)


def ensure_compatible_torch() -> None:
    """Kaggle's torch may drop older GPU archs (e.g. P100 sm_60). If the assigned
    GPU's arch isn't in the installed torch's build list, install an official wheel
    that includes it. The training/inference run as subprocesses, so they pick up
    the freshly-installed torch even though this process imported the old one."""
    import torch
    assert torch.cuda.is_available(), "no GPU on this kernel — enable the GPU accelerator"
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    name = torch.cuda.get_device_name(0)
    print(f"CUDA: {name} | capability {sm} | torch archs {arch_list}", flush=True)
    if any(sm in a for a in arch_list):
        print(f"torch supports {sm}", flush=True)
        return
    print(f"torch lacks {sm}; installing official cu121 torch trio with Pascal support", flush=True)
    # Downgrade torch + torchvision + torchaudio together: Kaggle pins them to its
    # own torch, so a lone torch downgrade leaves a broken torchvision (operator
    # torchvision::nms missing) that crashes the transformers import chain.
    sh("pip install -q torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "
       "--index-url https://download.pytorch.org/whl/cu121 "
       "--extra-index-url https://pypi.org/simple")
    # Kaggle's transformers (>=4.50) blocks torch.load unless torch>=2.6 (CVE-2025-
    # 32434). We're on torch 2.4.1 (for sm_60) and NLLB-600M has no safetensors, so
    # pin a pre-block transformers that loads the .bin via torch.load.
    sh("pip install -q 'transformers==4.46.3'")
    sh('python -c "import torch, torchvision, transformers; print(\'reinstalled\', '
       'torch.__version__, torchvision.__version__, transformers.__version__, '
       'torch.cuda.get_arch_list())"')


def main() -> None:
    # No -U: Kaggle pre-installs most of these. Requires internet (phone-verified
    # Kaggle account). COMET is computed separately (see DEPS note).
    sh("pip install -q " + " ".join(DEPS))
    stage_project()
    os.chdir(REPO)
    ensure_compatible_torch()

    # expandable_segments reduces CUDA fragmentation on the 16GB GPU.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if SMOKE:
        print("=== SMOKE MODE: tiny+fast end-to-end validation ===", flush=True)
    ft = " --smoke" if SMOKE else ""
    lim = f" --limit {SMOKE_LIMIT}" if SMOKE else ""
    suf = f"_n{SMOKE_LIMIT}" if SMOKE else ""

    sh(f"python src/train/finetune_nllb.py --config src/train/config.yaml --out_dir {CKPT}{ft}")
    sh(f"python src/infer/translate.py --split test --system nllb_zeroshot{lim}")
    sh(f"python src/infer/translate.py --split test --system nllb_finetuned --model {CKPT}/best{lim}")
    # chrF++/spBLEU/BLEU here; COMET is computed later from the pulled hyps.
    sh(f"python src/eval/automatic.py "
       f"--hyps experiments/results/nllb_zeroshot_test{suf}_hyps.txt "
       f"--refs data/processed/test.jsonl --system nllb_zeroshot{lim}")
    sh(f"python src/eval/automatic.py "
       f"--hyps experiments/results/nllb_finetuned_test{suf}_hyps.txt "
       f"--refs data/processed/test.jsonl --system nllb_finetuned{lim}")

    # --- FLORES+ cross-eval (out-of-domain); flores.jsonl is bundled in the dataset
    # and auto-staged by stage_project, so no (gated) HF download is needed. ---
    sh(f"python src/infer/translate.py --split flores --system nllb_zeroshot{lim}")
    sh(f"python src/infer/translate.py --split flores --system nllb_finetuned --model {CKPT}/best{lim}")
    sh(f"python src/eval/automatic.py "
       f"--hyps experiments/results/nllb_zeroshot_flores{suf}_hyps.txt "
       f"--refs data/processed/flores.jsonl --system nllb_zeroshot_flores{lim}")
    sh(f"python src/eval/automatic.py "
       f"--hyps experiments/results/nllb_finetuned_flores{suf}_hyps.txt "
       f"--refs data/processed/flores.jsonl --system nllb_finetuned_flores{lim}")

    sh("python src/eval/make_tables.py")

    for rel in ARTIFACTS:
        src = os.path.join(REPO, rel)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(WORK, os.path.basename(src)))
            print("saved ->", os.path.basename(src))

    # PERSIST THE FINE-TUNED CHECKPOINT (never leave a trained model in ephemeral /tmp).
    # Copied into /kaggle/working so it survives in the kernel output and can be attached
    # to a future kernel (kernel-output-as-input) for FLORES+/extra inference WITHOUT
    # re-training. Skipped in smoke mode (the tiny model is throwaway).
    best = os.path.join(CKPT, "best")
    if not SMOKE and os.path.isdir(best):
        dst = os.path.join(WORK, "ckpt_best")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(best, dst)
        print("saved checkpoint -> ckpt_best/ (persisted in kernel output)", flush=True)
    print("\nPIPELINE DONE")


if __name__ == "__main__":
    main()
