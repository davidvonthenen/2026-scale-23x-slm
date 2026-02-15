# Building An SQL-based SLM Using Fine-Tuning (and Quantization)

This project provides an end-to-end build of a SQL-based Small Language Model (SLM) via fine-tuning and quantization.

## Prerequisites

- Python 3.12+
- An H100 (or better)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Data Prep + Fine-Tune A Model (Qwen 2.5)

This step will perform data prep and fine-tune your model.

```bash
python 1_finetune.py
```

### Step 2: Quantize Your Fine-Tuned Model

Take the model you trained in step 1 and quantize it.

```bash
python 2_quantize.py
```

If you don't have access to that kind of hardware, download these prebuild models below.

PRIMARY DOWNLOAD:
https://drive.google.com/file/d/1FV_e5QaQuY6Qg3PDFfNIj4_ZyCS_y41l/view?usp=drive_link

BACKUP DOWNLOAD:
https://drive.google.com/file/d/1urW29pDRNoHHUnuwUiqDne2zpo221Spm/view?usp=drive_link

### Step 3: Run Inference On Your Model

Download the model from the previous step (if needed), the run inference on your model.

```bash
python 3_inference.py
```
