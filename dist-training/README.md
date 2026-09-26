# LoRA + Horovod Training Pipeline — Setup & Deployment Notes

**Audience for this doc:** whoever is setting up and running this pipeline
(currently expected to be the lead engineer). Every file below lists exactly
which machine it runs on, what has to already be true before running it, and
what it hands off to the next step.

## Overview of the pipeline

Raw articles → (RAG API call) → training CSV → (Horovod cluster) → trained
LoRA adapter file → handed to the RAG service. Two machine types are involved:
**one laptop/dev machine** (for writing and validating code before touching
real infrastructure) and **the VM cluster** (for the real distributed run).

---

## File-by-file reference

### `setup_laptop.sh`
- **Runs on:** Any single dev machine (does not need to be the VMs).
- **When:** First thing, before anything else. No prerequisites.
- **What it does:** Installs PyTorch + Horovod using the Gloo backend (no MPI,
  no multi-machine networking required) plus the ML libraries the training
  script needs.
- **Run once** per machine that will be used to test code before deploying to
  the VM cluster. Re-run only if the environment gets wiped.

### `test_horovod.py`
- **Runs on:** Both the dev machine (first) and every VM (later, as a
  cluster-verification step).
- **When (dev machine):** Immediately after `setup_laptop.sh`, before writing
  or trusting any training code.
- **When (VMs):** After `setup_vm.sh` has been run on every VM, and after the
  hostfile and SSH keys are configured — this is the first real test that the
  cluster itself works, *before* running real training.
- **What it does:** Each worker process prints its Horovod rank/size. If every
  expected worker prints a line, the cluster (or the local simulation) is
  wired correctly.
- **Dev machine command:** `horovodrun -np 2 --gloo python test_horovod.py`
- **VM cluster command:** `horovodrun -np 2 -hostfile hostfile python test_horovod.py`

### `train.py`
- **Runs on:** Dev machine first (for validation), then the VM cluster (for
  the real run). **This file does not change between the two** — same file,
  copied over unmodified.
- **When (dev machine):** After `test_horovod.py` passes locally. Use a dummy
  CSV (see "Placeholders" below) to confirm the training logic itself runs
  without errors — this is purely a code-correctness check, not real training.
- **When (VM cluster):** After `test_horovod.py` passes on the real cluster,
  AND `cleaned_dialect_news.csv` (the real training data — see
  `generate_training_data.py` below) is present on disk.
- **What it does:** Loads a frozen base LLM, attaches a LoRA adapter, trains
  that adapter (only) on the CSV, splitting the CSV across however many
  workers are running, keeping every worker's adapter synced via Horovod.
- **Produces:** `./lora_adapter_output/` — a small folder (a few MB), NOT a
  full model. This is the artifact that eventually goes to the RAG team.
- **Dev machine command:** `horovodrun -np 2 --gloo python train.py`
- **VM cluster command:** `horovodrun -np 2 -hostfile hostfile python train.py`

### `setup_vm.sh`
- **Runs on:** Every VM in the cluster, individually.
- **When:** Once the VMs exist and are reachable via SSH. No dependency on
  any other file in this repo.
- **What it does:** Installs NVIDIA driver + CUDA (skip this part if VMs are
  already provisioned with drivers — just confirm with `nvidia-smi`), Open
  MPI, and PyTorch + Horovod built for NCCL (the real multi-machine backend,
  different from the laptop's Gloo install).
- **Note:** The SSH keygen/copy-id lines at the bottom are commented out and
  should be run manually, once, only on whichever VM is designated "master."

### `hostfile`
- **Lives on:** The master VM only.
- **When:** Filled in after all VMs are provisioned and their IP addresses
  are known. Must exist before any `-hostfile` command is run.
- **What it is:** Not a script — a plain text list of VM IPs and how many GPU
  slots (GPUs) each one has. Must be edited with real IPs before use; the
  version in this repo has placeholder IPs.

### `generate_training_data.py`
- **Runs on:** Any machine with network access to the RAG team's API. Does
  NOT need a GPU and does NOT need to be a VM — this is a data-prep step, not
  training.
- **When:** Can run any time after the RAG team's API is available and its
  request/response format is known. Independent of VM setup — can happen in
  parallel with provisioning the cluster.
- **What it does:** Calls the RAG service's retrieval endpoint for each raw
  article, builds a `context` + `target` pair, writes them to
  `cleaned_dialect_news.csv`.
- **Produces:** `cleaned_dialect_news.csv` — this file must be copied onto
  every VM before `train.py` can run for real (see "Placeholders" — this file
  has hardcoded values that need real information from the RAG team first).

### `inference_with_rag.py`
- **Runs on:** Wherever the RAG service itself runs — NOT the training VMs,
  NOT the dev machine. This file is not meant to be executed as part of this
  pipeline; it's a reference snippet for whoever owns the RAG service's code.
- **When:** After `train.py` has produced `lora_adapter_output/` and that
  folder has been handed off (shared storage / S3 / however files move
  between services here) to the RAG service's environment.
- **What it does:** Shows the one-line change needed in the RAG service's
  existing model-loading code to use the trained adapter instead of the bare
  base model. Retrieval and prompt-building logic in the RAG service is
  unaffected.

---

## Full deployment sequence

1. `setup_laptop.sh` on a dev machine.
2. `test_horovod.py` locally (`--gloo`) — confirms Horovod works at all.
3. `train.py` locally (`--gloo`) with dummy data — confirms the training code
   itself is correct, isolated from any cluster/network issues.
4. `setup_vm.sh` on every VM.
5. Configure SSH keys (master → every worker) and fill in `hostfile` with
   real IPs.
6. `test_horovod.py` on the real cluster (`-hostfile`) — confirms the cluster
   itself is wired correctly.
7. `generate_training_data.py` (can happen any time after step 1, in
   parallel with steps 4–6, once RAG API details are known) →
   `cleaned_dialect_news.csv`.
8. Copy `cleaned_dialect_news.csv` onto every VM.
9. `train.py` on the real cluster (`-hostfile`) → produces
   `lora_adapter_output/`.
10. Hand `lora_adapter_output/` + `inference_with_rag.py` to whoever owns the
    RAG service.

## Placeholders that MUST be filled in with real values before this runs end to end

| File | Placeholder | Who provides the real value |
|---|---|---|
| `generate_training_data.py` | `RAG_API_URL`, response field name, how `raw_articles` is loaded | RAG team |
| `train.py` | `target_modules=["c_attn"]` (GPT-2-specific) | Whoever picks the real base model — run `print(model)` and match its actual attention layer names |
| `hostfile` | Placeholder IPs | Whoever provisions the VMs |
| `inference_with_rag.py` | `BASE_MODEL_NAME`, `ADAPTER_PATH` | Coordinated between this pipeline's owner and the RAG service's owner |

## Command reference (only difference between dev machine and VM cluster)

- Dev machine: `horovodrun -np 2 --gloo python <script>.py`
- VM cluster: `horovodrun -np 2 -hostfile hostfile python <script>.py`

The `.py` files themselves are never edited between the two environments —
only this one flag changes.
