# Hugging Face profile — final manual steps (no API for these)

Everything programmable is done: the model is published, weak repos privatized,
throwaways deleted, and a Collection created. These last two are account-settings
clicks (HF has no API for them). ~2 minutes.

## What's already done (automated)

- **Model published (public):** https://huggingface.co/PyaeSoneK/nllb-600m-wikihow-en-my
- **Collection (renders on your profile):** https://huggingface.co/collections/PyaeSoneK/english-to-burmese-mt-wikihow-6a28b3f136f02ab48e0d62c4
- **Privatized** (hidden from public profile, reversible): LlamaV2LegalFineTuned,
  finetuned_pythia-2.8b-deduped_legal, Fine_Tuned_Pythia_smallest_140_legal,
  pythia_70m_legalQA, LegalFewShot (dataset), legalQAcustom + chatchat (Spaces)
- **Deleted:** LegalQAfewshot, autotrain-data-legalqa, autotrain-data-legalqacustom,
  Trial_Seallm_with_mydata

---

## 1. Set your bio + links

Go to **https://huggingface.co/settings/account** → "Bio" / profile fields.

**Bio (option A — research-forward, recommended):**

```
AI Engineer & low-resource NLP researcher. Building English→Burmese machine
translation for instructional text (WikiHow-MY). Founder @ Ekkhara, Station F Paris.
```

**Bio (option B — shorter):**

```
Low-resource MT / Burmese NLP. English→Burmese translation for how-to text.
Founder @ Ekkhara (Station F, Paris).
```

**Links to fill in the profile fields:**

- Website: `https://pseonkyaw.dev`
- GitHub: `soneeee22000`
- (Google Scholar / X / LinkedIn: add if you want them shown)

## 2. Set status to "Open to Work"

Click your **avatar (top-right) → your status / "Set status"** → choose
**"Open to work"** (and optionally "Open to collaboration"). It shows a badge on
your avatar across the Hub — useful while job-hunting.

## 3. (Later, when the paper is on arXiv)

- On the model page → add the arXiv link under References so HF cross-links
  paper ↔ model.
- **Settings → Papers** → claim the paper → tick **"Show on profile."**
- **Mint DOIs** on the model (Settings → DOI → Generate) and on the corpus dataset
  when you publish it, so both are formally citable on your CV and in the paper.
- Publish the **WikiHow-MY corpus** as a dataset (rehydration form) and link it from
  the model's `datasets:` field — we deferred this until the paper is public.

## Optional polish

- Add a **profile avatar** if you don't have one (settings → avatar).
- Consider a tiny **Gradio Space demo** of the model so the "Try it" widget shows —
  high signal, makes the profile look alive.
