"""Kaggle GPU kernel entrypoint: MetricX-24 scoring of the benchmark systems.

Scores each system's hypotheses with MetricX-24 (reference-based hybrid, mT5),
which BURMESE-SAN uses, so our table is comparable on the same learned metric.
MetricX is not pip-installable (no package), so we git-clone google-research/metricx
and run metricx24.predict via PYTHONPATH. Inputs (one JSONL per system with
source/hypothesis/reference) ride in the attached private dataset, produced by
src/eval/build_metricx_inputs.py. Scores are mean error in [0,25] (LOWER is better).

Result -> /kaggle/working/metricx_results.json, merged locally into main_results.json.
Do not run locally; assumes the Kaggle filesystem + GPU.
"""
import glob
import json
import os
import shutil
import subprocess

WORK = "/kaggle/working"
METRICX = "/tmp/metricx"
INPUTS = "/tmp/metricx_inputs"
SMOKE = False  # flipped to True by `kaggle_metricx.py --smoke`
SMOKE_LIMIT = 8
SMOKE_SYSTEMS = ("nllb_zeroshot", "nllb_zeroshot_flores")  # one per ref split

TRANSFORMERS_PIN = "4.46.3"   # 4.x (5.x breaks predict.py); proven on this GPU stack
TOKENIZER = "google/mt5-xl"   # all hybrid sizes use the mt5-xl tokenizer
MODEL_FULL = "google/metricx-24-hybrid-xl-v2p6"
MODEL_SMOKE = "google/metricx-24-hybrid-large-v2p6"  # small/fast: validate the path cheaply
MAX_INPUT_LENGTH = 1536
BATCH_SIZE = 1
METRIC_KEY = "metricx24"


def sh(cmd: str, env: dict = None) -> None:
    """Run a shell command, streaming output, aborting the kernel on failure."""
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, env=env)


def ensure_compatible_torch() -> None:
    """Install a torch wheel that supports the assigned GPU arch if the pre-installed
    one doesn't (Kaggle's torch can drop Pascal P100 sm_60). Inference subprocesses
    pick up the freshly-installed torch."""
    import torch
    assert torch.cuda.is_available(), "no GPU on this kernel — enable the GPU accelerator"
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    print(f"CUDA: {torch.cuda.get_device_name(0)} | {sm} | archs {torch.cuda.get_arch_list()}",
          flush=True)
    if any(sm in a for a in torch.cuda.get_arch_list()):
        print(f"torch supports {sm}", flush=True)
        return
    print(f"torch lacks {sm}; installing cu121 torch trio with Pascal support", flush=True)
    sh("pip install -q torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "
       "--index-url https://download.pytorch.org/whl/cu121 "
       "--extra-index-url https://pypi.org/simple")


def stage_inputs() -> None:
    """Copy the per-system MetricX input JSONLs out of the mounted dataset."""
    os.makedirs(INPUTS, exist_ok=True)
    found = glob.glob("/kaggle/input/**/metricx_inputs/*.jsonl", recursive=True)
    if not found:  # fall back: any *.jsonl that looks like a metricx input
        found = glob.glob("/kaggle/input/**/*.jsonl", recursive=True)
    for path in found:
        shutil.copy(path, os.path.join(INPUTS, os.path.basename(path)))
    staged = sorted(os.listdir(INPUTS))
    print("staged inputs:", staged, flush=True)
    assert staged, "no metricx input jsonls found in /kaggle/input"


def limit_file(src: str, dst: str, n: int) -> None:
    """Write the first n lines of src to dst (smoke mode)."""
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if i >= n:
                break
            fo.write(line)


def score_system(system: str, model: str, env: dict) -> float:
    """Run metricx24.predict for one system; return mean prediction (lower=better)."""
    in_path = os.path.join(INPUTS, f"{system}.jsonl")
    if SMOKE:
        small = os.path.join(INPUTS, f"{system}.smoke.jsonl")
        limit_file(in_path, small, SMOKE_LIMIT)
        in_path = small
    out_path = os.path.join(INPUTS, f"{system}.pred.jsonl")
    sh(f"python -m metricx24.predict --tokenizer {TOKENIZER} "
       f"--model_name_or_path {model} --max_input_length {MAX_INPUT_LENGTH} "
       f"--batch_size {BATCH_SIZE} --input_file {in_path} --output_file {out_path}",
       env=env)
    preds = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            preds.append(float(json.loads(line)["prediction"]))
    score = round(sum(preds) / len(preds), 4)
    print(f"{system}: MetricX-24 {score} (n={len(preds)})", flush=True)
    return score


def main() -> None:
    sh(f"pip install -q 'transformers=={TRANSFORMERS_PIN}' sentencepiece accelerate protobuf")
    ensure_compatible_torch()
    sh(f"git clone --depth 1 https://github.com/google-research/metricx {METRICX}")
    stage_inputs()

    env = dict(os.environ)
    env["PYTHONPATH"] = METRICX + os.pathsep + env.get("PYTHONPATH", "")
    # predict.py uses transformers.Trainer, which auto-inits wandb on Kaggle (installed
    # but unauthenticated) and aborts. Disable all experiment-tracking integrations.
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    model = MODEL_SMOKE if SMOKE else MODEL_FULL
    systems = SMOKE_SYSTEMS if SMOKE else tuple(
        f[: -len(".jsonl")] for f in sorted(os.listdir(INPUTS))
        if f.endswith(".jsonl") and ".pred" not in f and ".smoke" not in f)
    if SMOKE:
        print("=== SMOKE MODE: large model, few rows, 2 systems ===", flush=True)

    results = {}
    for system in systems:
        results[system] = {METRIC_KEY: score_system(system, model, env),
                           f"{METRIC_KEY}_model": model}

    out = os.path.join(WORK, "metricx_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved -> metricx_results.json", flush=True)
    print("\nPIPELINE DONE")


if __name__ == "__main__":
    main()
