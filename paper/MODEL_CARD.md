---
language:
  - en
  - my
license: cc-by-nc-4.0
base_model: facebook/nllb-200-distilled-600M
library_name: transformers
pipeline_tag: translation
tags:
  - translation
  - nllb
  - burmese
  - wikihow
  - low-resource
datasets:
  - facebook/flores
metrics:
  - chrf
  - bleu
model-index:
  - name: nllb-600m-wikihow-en-my
    results:
      - task:
          type: translation
          name: Machine Translation (English to Burmese)
        dataset:
          name: FLORES-200 (devtest, eng_Latn-mya_Mymr)
          type: facebook/flores
          config: eng_Latn-mya_Mymr
          split: devtest
        metrics:
          - type: chrf
            name: chrF++
            value: 33.5
          - type: bleu
            name: spBLEU
            value: 17.78
      - task:
          type: translation
          name: Machine Translation (English to Burmese)
        dataset:
          name: WikiHow-MY (held-out test)
          type: wikihow
          split: test
        metrics:
          - type: chrf
            name: chrF++
            value: 41.64
          - type: bleu
            name: spBLEU
            value: 23.18
---

# NLLB-200-distilled-600M fine-tuned for English to Burmese (WikiHow-MY)

This is [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
fine-tuned for **English to Burmese (`eng_Latn` to `mya_Mymr`)** translation on an
instructional-text corpus derived from wikiHow. It targets the procedural /
how-to register, where the zero-shot NLLB baseline is weakest.

## Model description

- **Base model:** `facebook/nllb-200-distilled-600M` (600M-param distilled NLLB-200)
- **Architecture:** `M2M100ForConditionalGeneration` (seq2seq, SentencePiece tokenizer)
- **Direction:** English (`eng_Latn`) to Burmese (`mya_Mymr`), single direction
- **Fine-tuning data:** WikiHow-MY instructional EN to MY pairs (see Training data)
- **Selection metric:** dev **chrF** (more stable than BLEU on unsegmented Burmese)

## Intended uses & limitations

**Intended use.** Translating English instructional / how-to text into Burmese
(steps, tips, procedural prose). Research and non-commercial use only.

**Out of scope / limitations.**

- **Non-commercial only** — inherits CC-BY-NC-4.0 from NLLB-200 (see License).
- Tuned on a single domain (wikiHow). Expect degradation on conversational,
  legal, medical, or other out-of-domain text.
- Single direction (en to my). Do not use for my to en.
- Burmese has no orthographic word boundaries; downstream metrics and any
  word-level processing must account for this. Reported BLEU is **spBLEU**
  (SentencePiece-tokenized) and the primary metric is **chrF++**.
- May hallucinate, drop, or mistranslate named entities and numbers; not for
  high-stakes use without human review.

## Training data

Fine-tuned on **WikiHow-MY**, parallel English to Burmese instructional
sentence pairs extracted from wikiHow articles. wikiHow content is licensed
**CC-BY-NC-SA-3.0**; this derivative therefore carries a non-commercial,
share-alike obligation in addition to NLLB's CC-BY-NC (see License & attribution).

- Train / dev / test: **8,302 / 908 / 846** pairs (~10,056 total, from 82 articles)
- **Article-disjoint** splits (seed 42): no article appears in more than one split,
  so the benchmark has no train/test leakage at the article level.
- Preprocessing: Zawgyi-to-Unicode normalization, de-duplication, and
  sentence-level alignment of the English-Burmese instructional pairs.

## Training procedure

Fine-tuned with 🤗 Transformers `Seq2SeqTrainer`.

| Hyperparameter          | Value                             |
| ----------------------- | --------------------------------- |
| base model              | facebook/nllb-200-distilled-600M  |
| learning rate           | 3e-5                              |
| warmup ratio            | 0.05                              |
| epochs                  | up to 10 (early stop on dev chrF) |
| per-device train batch  | 4                                 |
| grad accumulation       | 8 (effective batch 32)            |
| precision               | fp16                              |
| eval / save steps       | 250                               |
| early-stopping patience | 4                                 |
| max length              | 256                               |
| seed                    | 42                                |
| beams (eval/inference)  | 5                                 |

## Evaluation results

Evaluated on FLORES-200 devtest (`eng_Latn-mya_Mymr`) and the held-out
WikiHow-MY test split. chrF++ and spBLEU computed with `sacrebleu`.

| System / test set                   | chrF++ | spBLEU | BLEU |
| ----------------------------------- | -----: | -----: | ---: |
| **This model** — WikiHow-MY test    |  41.64 |  23.18 | 3.74 |
| Base NLLB zero-shot — WikiHow-MY    |  36.01 |  19.33 | 2.62 |
| **This model** — FLORES-200 devtest |  33.50 |  17.78 | 2.71 |
| Base NLLB zero-shot — FLORES-200    |  29.17 |  14.79 | 2.19 |

Fine-tuning on WikiHow improves the procedural in-domain test by **+5.6 chrF++**
(36.01 to 41.64) and also lifts general-domain FLORES-200 by **+4.3 chrF++**
(29.17 to 33.50) — i.e. domain adaptation with no catastrophic forgetting.
chrF++ is the primary metric (segmentation-agnostic for unsegmented Burmese);
spBLEU uses the FLORES-200 tokenizer. Numbers from
`experiments/results/main_results.json`.

## How to use

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "PyaeSoneK/nllb-600m-wikihow-en-my"
tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = "Fold the paper in half, then crease the edge firmly."
inputs = tokenizer(text, return_tensors="pt")

# Force the decoder to start generating in Burmese.
bos = tokenizer.convert_tokens_to_ids("mya_Mymr")
generated = model.generate(
    **inputs,
    forced_bos_token_id=bos,
    num_beams=5,
    max_new_tokens=256,
)
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

Note: on recent `transformers`, prefer
`tokenizer.convert_tokens_to_ids("mya_Mymr")` over the deprecated
`tokenizer.lang_code_to_id[...]`.

## License & attribution

This model is released under **CC-BY-NC-4.0**. Two upstream non-commercial
terms apply and you must comply with both:

1. **NLLB-200** (`facebook/nllb-200-distilled-600M`) is **CC-BY-NC-4.0**. Any
   derivative — including this fine-tune — must remain non-commercial and
   attribute Meta AI / the NLLB Team.
2. **wikiHow** training content is **CC-BY-NC-SA-3.0**: non-commercial **and
   share-alike**. Reuse of this model or its outputs must credit wikiHow and
   carry a compatible non-commercial license.

Effective terms = the union of these: **non-commercial use only**, attribution
to both NLLB and wikiHow, and share-alike where wikiHow-derived content is
redistributed.

## Citation

```bibtex
@misc{nllb600m_wikihow_en_my,
  title  = {NLLB-200-distilled-600M fine-tuned for English-Burmese on WikiHow-MY},
  author = {Pyae Sone Kyaw},
  year   = {2026},
  note   = {Fine-tune of facebook/nllb-200-distilled-600M, CC-BY-NC-4.0}
}

@article{nllb2022,
  title   = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author  = {{NLLB Team} and Costa-juss\`a, Marta R. and others},
  journal = {arXiv preprint arXiv:2207.04672},
  year    = {2022}
}
```
