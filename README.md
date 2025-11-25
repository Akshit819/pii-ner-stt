# PII NER Assignment Skeleton
# 🔐 PII Entity Recognition for Noisy STT Transcripts  
**High-Precision, Low-Latency Token Classification Model**

This repository contains my submission for the **IIT Bombay — PII NER for Noisy STT Transcripts** assignment.  
The goal was to build a **token-level NER system** that detects PII entities from noisy speech-to-text transcripts with **high precision** and **low CPU latency**.
--

# 📘 **1. Project Overview**

The system identifies these entity types:

- **PII (PII = true):**  
  `CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`  
- **Non-PII (PII = false):**  
  `CITY`, `LOCATION`  

Outputs are **character-level spans** based on the original noisy transcript.

Priority goals:

- ✔ **High precision on PII types**  
- ✔ **Latency under 20ms on CPU (batch size = 1)**  
- ✔ Robustness to noisy STT patterns (“dot”, “at”, missing punctuation, spacing errors)

---

# 📂 **2. Repository Structure**


This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
# Dev set
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

# (Optional) stress test set
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

Your task in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).
# pii-ner-stt
# pii-ner-stt
