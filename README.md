# Qwen2.5-0.5B Medical LoRA Fine-Tuning Tutorial
## My Reproduction and Extension

I reproduced the LoRA fine-tuning pipeline for Qwen2.5-0.5B-Instruct on a Chinese medical question-answering dataset.

I further conducted a LoRA rank ablation study with r = 4, 8, and 16 to compare parameter efficiency and validation performance.

| Rank | Trainable Parameters | Train Loss | Eval Loss |
|---|---:|---:|---:|
| 4 | 2,199,552 | 3.009 | 2.717 |
| 8 | 4,399,104 | 2.944 | 2.649 |
| 16 | 8,798,208 | 2.941 | 2.567 |

I also compared the base model and the LoRA fine-tuned model on five medical questions. The fine-tuned model showed stronger domain-specific response characteristics, while some outputs also revealed repetition and potentially unreliable advice, highlighting the importance of data quality and model evaluation.

A hands-on learning project for LLM fine-tuning. 7 modules covering the full pipeline: data processing, SFT training, inference comparison, and ablation experiments.

For people with Python and PyTorch experience who want to learn LoRA fine-tuning by doing.

[中文版 README](README_CN.md)

## What This Project Does

Fine-tunes Qwen2.5-0.5B-Instruct (498M parameters) with LoRA SFT on 640 Chinese medical Q&A samples, then compares base vs. fine-tuned model outputs.

```
Raw data → Clean → SFT preprocessing → LoRA training → Inference → Evaluation → Ablation
 Module 1    Module 1       Module 2         Modules 3-4    Module 5    Module 6     Module 7
```

## 7 Modules

| Module | File | What You Learn |
|--------|------|----------------|
| 1. Data Preparation | `scripts/my_prepare_data.py` | jsonl I/O, data cleaning, use original train/valid/test split |
| 2. SFT Preprocessing | `scripts/my_train_lora.py` | Messages format, tokenization, assistant-only loss mask (-100) |
| 3. LoRA Config | `scripts/my_train_lora.py` | LoRA theory, get_peft_model, target_modules |
| 4. Training | `scripts/my_train_lora.py` | TrainingArguments, Trainer, gradient accumulation, bf16 |
| 5. Inference Comparison | `scripts/my_infer_compare.py` | PeftModel loading, sampling generation, VRAM management |
| 6. Batch Evaluation | `scripts/my_eval_lora.py` | Batch inference, saving results to jsonl |
| 7. Ablation Experiments | Same script as Module 4 | Comparing different ranks and data sizes |

Each module maps to a `my_*.py` file in `scripts/`, with a reference implementation (`scripts/train_lora.py` etc.) alongside it for comparison.

## Key Concepts

### SFT Data Processing

```python
# Raw data
{"instruction": "...", "input": "How to treat hypertension?", "output": "Patients should..."}

# Convert to messages format
[
    {"role": "system",    "content": "You are a medical Q&A assistant."},
    {"role": "user",      "content": "Answer the following medical question...\n\nQuestion: How to treat hypertension?"},
    {"role": "assistant", "content": "Patients should..."},
]

# After tokenization, construct labels (assistant-only loss mask)
# input_ids: [prompt tokens...] + [answer tokens...]
# labels:    [-100, -100, ...]    + [answer tokens...]
# → Loss is only computed on the answer portion
```

### LoRA Theory

```
h = Wx + (alpha/r) * BAx

W: original weights (frozen)
A: d x r matrix (trained)
B: r x d matrix (trained)
alpha/r: scaling factor

This project: r=8, alpha=16, 4.4M trainable params (0.88% of total)
```

### Ablation Results

| Experiment | Finding |
|------------|---------|
| rank 4/8/16 comparison | Nearly identical loss (~2.764), bottleneck is data not capacity |
| 200 vs 640 samples | More data lowers train_loss (2.823 -> 2.720), but gains are limited |

## Dataset

Uses [shibing624/medical](https://huggingface.co/datasets/shibing624/medical), a Chinese medical Q&A dataset with instruction, input, and output fields.

The original data contains 8000+ Chinese medical Q&A pairs covering hypertension, diabetes, common cold, insomnia, and more.

Data example:

```json
{
    "instruction": "Please answer the following medical health question in a careful, professional, and easy-to-understand manner.",
    "input": "What should be done about hypertension?",
    "output": "Hypertension patients should control salt intake, maintain regular exercise, monitor blood pressure regularly..."
}
```

The original dataset has train/valid/test splits. This project uses the original split directly.

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.1+ (recommended: 2.6+cu124)
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 / 4050 or above)
- ~3GB disk space (model + data)

### Installation

**Note:** This repo only contains code and scripts. You need to download the model and generate data yourself (~3GB total).

```bash
# 1. Clone and create environment
git clone https://github.com/SoloCalm/MiniLoRA.git
cd MiniLoRA
conda create -n minillm python=3.10
conda activate minillm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model
# Download Qwen2.5-0.5B-Instruct to models/ directory
# Or set model_name to the remote path (auto-downloads)
# Model URL: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

# 4. Download dataset
python download_dataset.py
# Raw data saved to data/medical_raw/finetune/

# For users in China, use HF mirror for faster downloads:
# HF_ENDPOINT=https://hf-mirror.com python download_dataset.py
```

### Run All 7 Modules

```bash
# Module 1: Data preparation (generates data/medical/*.jsonl)
python scripts/my_prepare_data.py

# Modules 2-4: Training (quick test with 50 samples)
python scripts/my_train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1 --grad-accum 4 --max-length 256

# Module 5: Single question inference comparison
python scripts/my_infer_compare.py --question "What should hypertension patients pay attention to in daily life?"

# Module 6: Batch evaluation
python scripts/my_eval_lora.py

# Module 7: Ablation experiments (compare different ranks)
python scripts/my_train_lora.py --r 4 --output-dir outputs/lora-r4 --max-train-samples 200 --epochs 1
python scripts/my_train_lora.py --r 8 --output-dir outputs/lora-r8 --max-train-samples 200 --epochs 1
python scripts/my_train_lora.py --r 16 --output-dir outputs/lora-r16 --max-train-samples 200 --epochs 1
```

## Project Structure

```
MiniLoRA/
├── README.md                           # This file (English)
├── README_CN.md                        # Chinese README
├── requirements.txt                    # Python dependencies
├── download_dataset.py                 # Dataset download script
├── scripts/
│   ├── prepare_medical_sft.py          # [Reference] Data preparation
│   ├── train_lora.py                   # [Reference] LoRA training
│   ├── infer_compare.py                # [Reference] Inference comparison
│   ├── my_prepare_data.py              # [Student] Module 1: Data preparation
│   ├── my_train_lora.py                # [Student] Modules 2-4: Preprocessing + LoRA + Training
│   ├── my_infer_compare.py             # [Student] Module 5: Inference comparison
│   └── my_eval_lora.py                 # [Student] Module 6: Batch evaluation
├── data/
│   ├── medical/
│   │   ├── train.jsonl                 # Training set (640 samples)
│   │   ├── valid.jsonl                 # Validation set (160 samples)
│   │   ├── test.jsonl                  # Test set (200 samples)
│   │   └── eval_prompts.jsonl          # Evaluation questions (10)
│   └── medical_raw/                    # Raw downloaded data
├── outputs/                            # LoRA adapter outputs
│   ├── lora-r4-1000/                   # rank=4 experiment
│   ├── lora-r8-1000/                   # rank=8 experiment
│   └── ...
└── results/
    ├── base_vs_lora.jsonl              # Base vs LoRA comparison results
    └── lora_ablation_summary.csv       # Ablation experiment summary
```

### About the scripts/ Directory

- `scripts/xxx.py` (no `my_` prefix): Reference code, pre-written and runnable
- `scripts/my_xxx.py` (with `my_` prefix): Blank templates with TODO hints for you to fill in

Learning flow: read the reference code to understand the logic, then close it and fill in the `TODO` sections in `my_*.py`.

## Code Walkthrough

### Module 1: Data Preparation (`my_prepare_data.py`)

3 functions:

```python
read_jsonl(path)                    # Read jsonl file
normalize_item(item)                # Clean a data item (unify instruction, filter empty output)
write_jsonl(path, items)            # Write jsonl (ensure_ascii=False)
```

Reads train/valid/test raw files separately, applies cleaning, and uses the original split directly.

Key point: `ensure_ascii=False` ensures Chinese characters are not escaped to `\uXXXX`.

### Modules 2-4: Training (`my_train_lora.py`)

```python
build_messages(example)              # Convert to messages format
preprocess(example, tokenizer, max_length)  # Tokenize + construct labels
load_model_and_lora(model_name, r)   # Load model + LoRA
main()                               # Full training pipeline
```

Key points:
- `apply_chat_template` return type varies; use `hasattr` to check
- Prompt portion of labels set to -100 (CrossEntropyLoss ignore_index)
- `DataCollatorForSeq2Seq` handles dynamic padding; pad positions get -100 in labels
- `remove_unused_columns=False` prevents Trainer from dropping custom columns

### Module 5: Inference Comparison (`my_infer_compare.py`)

```python
load_model(model_name, adapter_dir)  # Load base or LoRA model
generate(tokenizer, model, question) # Generate answer
```

Key points:
- 6GB VRAM can't hold two models at once; must `del model` + `gc.collect()` + `torch.cuda.empty_cache()`
- `temperature=0.7` controls randomness, `top_p=0.9` for nucleus sampling
- `torch.no_grad()` disables gradient computation, saving VRAM and speeding up inference

### Module 6: Batch Evaluation (`my_eval_lora.py`)

`eval_prompts.jsonl` is included in the repo (10 medical questions for evaluation).

Pipeline: read eval questions -> run base model on all -> free VRAM -> run LoRA model on all -> save comparison results.

### Module 7: Ablation Experiments

Use `--r` to set different LoRA ranks, `--max-train-samples` to control data size. Train each configuration and record loss.

## Future Directions

- [ ] Scale up training data (10k+ samples)
- [ ] Use larger models (Qwen2.5-7B)
- [ ] Add BLEU/ROUGE evaluation metrics
- [ ] Use GPT-4 for generation quality scoring
- [ ] Deploy inference service (vLLM / TGI)
- [ ] Add DPO/RLHF preference alignment
- [ ] Build a RAG (Retrieval-Augmented Generation) pipeline

## Dependencies

```
torch>=2.1.0
transformers>=4.45.0
datasets>=2.19.0
peft>=0.12.0
accelerate>=0.33.0
huggingface_hub>=0.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```
