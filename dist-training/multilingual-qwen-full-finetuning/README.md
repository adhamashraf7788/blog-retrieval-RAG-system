# Distributed Multilingual News Summarization — Qwen, No LoRA

This project demonstrates distributed **full fine-tuning** of
`Qwen/Qwen2.5-1.5B-Instruct` for news summarization in Arabic, English, and French.
It does not target regional dialect adaptation and does not use LoRA or PEFT.

## Why Qwen instead of GPT-2?

Qwen2.5-1.5B-Instruct is an instruction-tuned, multilingual causal language model
whose model card lists Arabic, English, and French among its supported languages.
It is a much more suitable starting point for this task than GPT-2. The model has
about 1.54 billion parameters.

Model page: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

## Architecture

Spark prepares article data → Parquet stores it → Petastorm reads and shards
records → Horovod synchronizes gradients across workers → PyTorch Transformers
trains all model parameters → the complete fine-tuned model is saved and used
by the RAG inference service.

## What makes this different from LoRA?

- **LoRA:** most base-model weights are frozen; small adapter matrices are trained.
- **This project:** all Qwen model parameters have `requires_grad=True`; Horovod
  synchronizes gradients for the full model; the checkpoint contains the full model.
- **Trade-off:** full fine-tuning uses much more memory, communication bandwidth,
  and compute than LoRA. The cluster must have enough RAM/VRAM to load the model,
  optimizer states, gradients, and activations on each worker.

## Languages

- Arabic: clear Modern Standard Arabic (MSA), with no regional dialect objective.
- English.
- French.

Training data should contain examples from all three languages. The Parquet dataset
must have `context` and `target` columns. `context` is the article/retrieved text;
`target` is the desired summary. Use article-level train/validation separation and
evaluate each language separately.

## Important hardware note

The default model is now Qwen2.5-1.5B-Instruct, not GPT-2. This is not a lightweight
CPU training workload. Distributed full fine-tuning of 1.5B parameters is generally
best done on compatible GPUs with sufficient memory and fast interconnects. A CPU
cluster can run the code in principle but may be extremely slow and require substantial
RAM. For a first test, use a tiny dataset, short sequences, and a small worker count.

The current training loop is a baseline implementation. For a robust multi-GPU run,
add mixed precision, gradient accumulation, gradient checkpointing, distributed
validation, checkpoint-resume, and rank-safe data handling. Confirm Petastorm sharding
and the shared filesystem configuration for your cluster before a long run.

## Setup

Install dependencies in a compatible environment:

```bash
pip install -r requirements-full-finetuning.txt
```

All workers must use compatible Python, PyTorch, Transformers, Horovod, and CUDA
versions (if using GPUs), and have access to the dataset.

## Configure (PowerShell)

```powershell
$env:MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
$env:DATA_PATH = "file:///path/to/multilingual_news.parquet"
$env:OUTPUT_DIR = "./full_model_output"
$env:MAX_LENGTH = "256"
$env:BATCH_SIZE = "1"
$env:EPOCHS = "1"
$env:LEARNING_RATE = "2e-5"
```

The model will download from Hugging Face on first use, so workers need access to
the model or a shared/local cache. For restricted network environments, pre-download
the model and set `MODEL_NAME` to its local directory.

## Run

A two-worker local smoke test, if Horovod/Gloo is correctly installed:

```powershell
horovodrun -np 2 --gloo python scripts/train.py
```

Multi-machine training additionally requires a valid hostfile, open worker-to-worker
network connectivity, and shared access to the Parquet data.

## Inference

`inference/inference_with_rag.py` loads the complete checkpoint from `full_model_output`
and uses Qwen's chat template to request same-language summarization. Retrieval itself
remains upstream; pass the retrieved context to `generate_from_context()`.

## Evaluation checklist

- Verify language balance and data quality.
- Keep train and validation articles separate.
- Report ROUGE or another suitable summarization metric per language, plus human checks.
- Check factual consistency and that summaries stay in the input language.
- Compare one worker and multiple workers for loss/throughput.
- Do not claim a successful distributed run until it has actually been tested.
