#!/bin/bash
# ============================================================
# VM SETUP — run once on EVERY VM. Debian-based. CPU-ONLY.
# Uses Gloo + Open MPI for multi-machine communication — no
# GPU driver, no CUDA, no NCCL. If GPUs are added later, see
# the note at the bottom of this file for what changes.
# ============================================================

# --- 1. Open MPI (process coordination across machines) ---
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev

# --- 2. PyTorch, CPU-only build (smaller download, no unused CUDA libs) ---
pip install torch --index-url https://download.pytorch.org/whl/cpu

# --- 3. Horovod, built WITHOUT any GPU backend ---
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch] --no-cache-dir
pip install peft transformers datasets accelerate pandas requests petastorm pyarrow

# Verify — PyTorch and Gloo should show enabled; NCCL will correctly show
# disabled here, that's expected and fine for CPU-only.
horovodrun --check-build

# --- 4. Passwordless SSH (run only on the MASTER VM, pointing at each worker) ---
# ssh-keygen -t rsa -b 4096
# ssh-copy-id user@worker-vm-ip

echo "Setup complete. Create a hostfile on the master VM before running training."

# ------------------------------------------------------------
# IF GPUs ARE ADDED LATER:
# - Re-add NVIDIA driver + CUDA toolkit install
# - Reinstall PyTorch from the matching cu1xx index URL instead of /cpu
# - Rebuild Horovod with HOROVOD_GPU_OPERATIONS=NCCL set
# - bitsandbytes becomes usable again for 4-bit quantization (CUDA-only;
#   intentionally left out of this CPU-only install above)
# ------------------------------------------------------------
