# Dialect LoRA Training Pipeline

LoRA + Horovod distributed training for regional dialect news, reading data
directly from Spark's Parquet output. No federated learning — this project
uses centralized distributed training only (see project history for why).

## Project structure

```
dialect-lora-pipeline/
├── README.md                    
├── setup/
│   ├── setup_laptop.sh          ← dev machine, one-time install
│   └── setup_vm.sh              ← every VM, one-time install
├── config/
│   └── hostfile                 ← master VM only, real IPs go here
├── scripts/
│   ├── generate_training_data.py ← reads Spark's Parquet, calls RAG API, writes training Parquet
│   ├── test_horovod.py          ← cluster smoke test, run before train.py
│   └── train.py                 ← the actual LoRA training job
└── inference/
    └── inference_with_rag.py    ← reference for the RAG service team, not run here
```


## Pipeline overview

```
Spark (cleans raw WARC crawl)
        ↓ writes Parquet
generate_training_data.py (calls RAG API, builds context/target pairs)
        ↓ writes training Parquet
train.py on the Horovod/VM cluster (Petastorm reads Parquet, LoRA trains, Horovod syncs workers)
        ↓ produces
lora_adapter_output/  (a few MB)
        ↓ handed off to
RAG service (inference_with_rag.py shows the one-line change needed there)
```


## File-by-file reference

### `setup/setup_laptop.sh`
- **Runs on:** Any single dev machine (does not need to be the VMs).
- **When:** First thing, before anything else. No prerequisites.
- **What it does:** Installs PyTorch + Horovod using the Gloo backend (no
  MPI, no multi-machine networking required) plus the ML/data libraries the
  scripts need (including `petastorm` and `pyarrow` for Parquet support).
- **Run once** per dev machine. Re-run only if the environment gets wiped.

### `setup/setup_vm.sh`
- **Runs on:** Every VM in the cluster, individually.
- **When:** Once the VMs exist and are reachable via SSH. No dependency on
  any other file in this repo.
- **What it does:** Installs NVIDIA driver + CUDA (skip if VMs are already
  provisioned with drivers — just confirm with `nvidia-smi`), Open MPI, and
  PyTorch + Horovod built for NCCL (the real multi-machine backend,
  different from the dev machine's Gloo install).
- **Note:** The SSH keygen/copy-id lines near the bottom are commented out
  and should be run manually, once, only on whichever VM is designated
  "master."

### `config/hostfile`
- **Lives on:** The master VM only.
- **When:** Filled in after all VMs are provisioned and their IP addresses
  are known. Must exist before any `-hostfile` command is run.
- **What it is:** Not a script — a plain text list of VM IPs and how many
  GPU slots (GPUs) each one has. The version in this repo has placeholder
  IPs and must be edited with real ones.

### `scripts/generate_training_data.py`
- **Runs on:** Any machine with network access to both the RAG service's
  API and wherever Spark's output lives. Does NOT need a GPU and does NOT
  need to be a VM — this is a data-prep step, not training.
- **When:** Any time after the RAG team's API is available and Spark's
  output path is known. Independent of VM setup — can happen in parallel
  with provisioning the cluster.
- **What it does:** Reads Spark's cleaned Parquet output, calls the RAG
  service's retrieval endpoint for each article, builds a `context` +
  `target` pair, writes them to `cleaned_dialect_news.parquet`.
- **Produces:** `cleaned_dialect_news.parquet` — must be copied onto every
  VM (or placed at a shared path every VM can reach) before `train.py` can
  run for real.

### `scripts/test_horovod.py`
- **Runs on:** Both the dev machine (first) and every VM (later, as a
  cluster-verification step).
- **When (dev machine):** Immediately after `setup_laptop.sh`, before
  writing or trusting any training code.
- **When (VMs):** After `setup_vm.sh` has run on every VM, and after the
  hostfile and SSH keys are configured — the first real test that the
  cluster itself works, before running real training.
- **What it does:** Each worker process prints its Horovod rank/size. If
  every expected worker prints a line, the cluster (or local simulation) is
  wired correctly.
- **Dev machine command:** `horovodrun -np 2 --gloo python scripts/test_horovod.py`
- **VM cluster command:** `horovodrun -np 2 -hostfile config/hostfile python scripts/test_horovod.py`

### `scripts/train.py`
- **Runs on:** Dev machine first (for validation), then the VM cluster (for
  the real run). **This file does not change between the two** — same
  file, copied over unmodified.
- **When (dev machine):** After `test_horovod.py` passes locally. Use a
  small dummy Parquet file (see "Placeholders" below) to confirm the
  training logic runs without errors — a code-correctness check, not real
  training.
- **When (VM cluster):** After `test_horovod.py` passes on the real
  cluster, AND `cleaned_dialect_news.parquet` is present at `PARQUET_PATH`
  on every VM.
- **What it does:** Loads a frozen base LLM, attaches a LoRA adapter, reads
  training data directly from Parquet via Petastorm, trains the adapter
  (only), with Petastorm sharding data across workers and Horovod keeping
  every worker's adapter synced via gradient averaging each step.
- **Produces:** `lora_adapter_output/` — a small folder (a few MB), NOT a
  full model.
- **Dev machine command:** `horovodrun -np 2 --gloo python scripts/train.py`
- **VM cluster command:** `horovodrun -np 2 -hostfile config/hostfile python scripts/train.py`

### `inference/inference_with_rag.py`
- **Runs on:** Wherever the RAG service itself runs — NOT the training
  VMs, NOT the dev machine. Not meant to be executed as part of this
  pipeline; it's a reference snippet for whoever owns the RAG service's
  code.
- **When:** After `train.py` has produced `lora_adapter_output/` and that
  folder has been handed off (shared storage / S3 / however files move
  between services here) to the RAG service's environment.
- **What it does:** Shows the one-line change needed in the RAG service's
  existing model-loading code to use the trained adapter instead of the
  bare base model. Retrieval and prompt-building logic in the RAG service
  is unaffected.

## Full deployment sequence

1. `setup/setup_laptop.sh` on a dev machine.
2. `test_horovod.py` locally (`--gloo`) — confirms Horovod works at all.
3. `train.py` locally (`--gloo`) with a dummy Parquet file — confirms the
   training code itself is correct, isolated from cluster/network issues.
4. `setup/setup_vm.sh` on every VM.
5. Configure SSH keys (master → every worker) and fill in `config/hostfile`
   with real IPs.
6. `test_horovod.py` on the real cluster (`-hostfile`) — confirms the
   cluster itself is wired correctly.
7. `generate_training_data.py` (can happen any time after step 1, in
   parallel with steps 4–6, once Spark's output path and RAG API details
   are known) → `cleaned_dialect_news.parquet`.
8. Copy `cleaned_dialect_news.parquet` onto every VM (or place at a shared
   path every VM can reach).
9. `train.py` on the real cluster (`-hostfile`) → produces
   `lora_adapter_output/`.
10. Hand `lora_adapter_output/` + `inference/inference_with_rag.py` to
    whoever owns the RAG service.

## Placeholders that MUST be filled in with real values before this runs end to end

| File | Placeholder | Who provides the real value |
|---|---|---|
| `scripts/generate_training_data.py` | `RAG_API_URL`, `SPARK_OUTPUT_PATH`, response field name | RAG team + Spark team |
| `scripts/train.py` | `target_modules=["c_attn"]` (GPT-2-specific), `PARQUET_PATH` | Whoever picks the real base model (layer names); Spark team (output path) |
| `config/hostfile` | Placeholder IPs | Whoever provisions the VMs |
| `inference/inference_with_rag.py` | `BASE_MODEL_NAME`, `ADAPTER_PATH` | Coordinated between this pipeline's owner and the RAG service's owner |

## Command reference (only difference between dev machine and VM cluster)

- Dev machine: `horovodrun -np 2 --gloo python scripts/<script>.py`
- VM cluster: `horovodrun -np 2 -hostfile config/hostfile python scripts/<script>.py`

