# Medical Q&A Retriever (MLE Screening Assignment)

This repository contains a compact, reproducible baseline solution for the medical assistant chatbot assignment.

## Problem Statement
Develop a medical chat system using the provided dataset of medical Q&A. The goal is to **generate answers** to user queries related to medical diseases. You may augment the dataset if desired (not required for this baseline).

---

## Approach

### 1) Data Preprocessing
- Removed nulls in `question` or `answer` (already applied to the provided dataset).
- Normalized text (lowercased + whitespace normalization).
- Deduplicated identical questions (based on normalized form).
- Split into **train/val/test** (80/10/10).

### 2) Model
A TF–IDF + Nearest Neighbors retriever (default k=3) with a small confidence gate:
- Vectorize training **questions** using TF–IDF with 1–2 grams.
- Retrieve top-k similar training questions for an incoming query using **cosine similarity**.
- **Generation**: return the **majority-vote** among the top-k retrieved answers (fallback to top-1). This acts as an extractive generator and is fast + robust.

Design rationale:
- Reliable baseline with limited compute and no external model downloads.
- Transparent evidence via retrieved Q–A examples.
- Easy to iterate into a **RAG** setup or fine-tuned seq2seq later.

### 3) Evaluation
We report:
- **BLEU** (1–4 gram overlap with smoothing)
- **ROUGE-L** (LCS-based F-measure)
- **TF-IDF cosine** similarity between predicted and gold answers

> Caveat: lexical metrics can under-represent semantic quality for paraphrases. In future work, prefer semantic metrics like BERTScore if allowed.

---

## Results

**Validation metrics**
```json
{
  "bleu": 0.11169862933095641,
  "rouge_l": 0.22491244824798204,
  "tfidf_cosine": 0.4613842888727
}
```

**Test metrics**
```json
{
  "bleu": 0.09731151463361018,
  "rouge_l": 0.21228514334520415,
  "tfidf_cosine": 0.45727322751424193
}
```

We also include three example interactions sampled from the test set in `examples.json`.

---

## Project Structure

```
medical_assistant_bot/
├── requirements.txt
├── metrics.json
├── examples.json
├── model.pkl
├── data_splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── src/
    ├── app.py
    ├── medical_bot.py
    └── metrics.py
```

---



## Assumptions
- The dataset contains reasonably paraphrased questions for the same medical facts.
- Extractive “generation” via retrieved answers is acceptable as a baseline.
- No external knowledge sources or model downloads are required by default.

---

## Limitations & Improvements
- **Paraphrases**: Consider semantic embeddings (e.g., Sentence-BERT) to improve retrieval.
- **RAG**: Add a trusted medical source (MedlinePlus, CDC) as a knowledge base and cite sections.
- **Abstractive generation**: Fine-tune a small seq2seq model (e.g., T5-small) with supervised pairs or use LoRA adapters if compute allows.
- **Evaluation**: Add semantic metrics (BERTScore) and human evaluation for clinical helpfulness + safety.

---

## Example Interactions
See `examples.json` for three held-out test examples with predicted answers and supporting evidence (top-1 shown per item).

---

## Reproducibility
- Random seed fixed for splitting.
- All artifacts pinned in this folder.
- Model serialized via pickle.

---

## Contact
For submission, upload this folder to GitHub and email the link as instructed.


---
_Last updated: 2025-08-28_

**Note**: If the top similarity falls below a threshold, the model prefers not to answer.


## Design notes
- Retrieval over generation: this dataset maps questions to reference answers; a lightweight retriever is a sensible first step.
- Confidence gating: returning nothing on very low similarity is preferable to hallucination.
- Evidence transparency: we always surface the top retrieved Q&A for inspection.

## How we tested
- Basic unit tests for metrics and retrieval (`pytest`).
- CI workflow runs tests on every push/PR.
- `diagnose` command provides quick dataset sanity checks.

_Last updated: 2025-08-28_
