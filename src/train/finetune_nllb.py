"""Fine-tune NLLB-200-distilled-600M en->my on WikiHow-MY. Runs on Colab GPU.

Reads src/train/config.yaml. Model selection on dev chrF (stable for Myanmar).
Saves the best checkpoint and writes test hypotheses for the eval harness.

Colab:
  !python src/train/finetune_nllb.py --config src/train/config.yaml --out_dir /content/ckpt
"""
import argparse
import json
import os

import numpy as np
import sacrebleu
import yaml
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return Dataset.from_list([{"en": r["en"], "my": r["my"]} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/train/config.yaml")
    ap.add_argument("--out_dir", default="ckpt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tcfg, gcfg = cfg["train"], cfg["generation"]

    tok = AutoTokenizer.from_pretrained(cfg["model"], src_lang=cfg["src_lang"],
                                        tgt_lang=cfg["tgt_lang"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"])

    def preprocess(batch):
        enc = tok(batch["en"], text_target=batch["my"],
                  max_length=tcfg["max_length"], truncation=True)
        return enc

    ds = {k: load_jsonl(cfg["data"][k]).map(
        preprocess, batched=True, remove_columns=["en", "my"])
        for k in ("train", "dev")}

    chrf = sacrebleu.CHRF(word_order=2)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dp = tok.batch_decode(preds, skip_special_tokens=True)
        dl = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf": chrf.corpus_score(dp, [dl]).score}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=tcfg["warmup_ratio"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        fp16=tcfg["fp16"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        predict_with_generate=True,
        generation_num_beams=gcfg["num_beams"],
        generation_max_length=gcfg["max_new_tokens"],
        seed=tcfg["seed"],
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["dev"],
        tokenizer=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=tcfg["early_stopping_patience"])],
    )
    trainer.train()
    best = os.path.join(args.out_dir, "best")
    trainer.save_model(best)
    tok.save_pretrained(best)
    print(f"best model -> {best}")
    print("Next: python src/infer/translate.py --split test "
          f"--system nllb_finetuned --model {best}")


if __name__ == "__main__":
    main()
