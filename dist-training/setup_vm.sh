#!/bin/bash
# ============================================================
# VM SETUP — run once on EVERY VM. Debian-based.
# Uses NCCL + Open MPI for real multi-machine communication.
# ============================================================

# --- 1. NVIDIA driver + CUDA (skip if your lead engineer already set this up —
#         just run `nvidia-smi` to confirm it works first) ---
sudo apt update
sudo apt install -y software-properties-common
# Manually edit /etc/apt/sources.list to add "contrib non-free non-free-firmware"
# to the end of each "deb" line, then:
sudo apt update
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1 nvidia-driver
echo ">>> REBOOT NOW, then continue with the rest of this script <<<"
# sudo reboot

# --- 2. Confirm GPU is visible ---
nvidia-smi

# --- 3. Open MPI ---
sudo apt install -y openmpi-bin libopenmpi-dev

# --- 4. PyTorch + Horovod, built for NCCL ---
pip install torch --index-url https://download.pytorch.org/whl/cu121
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch] --no-cache-dir
pip install peft transformers datasets accelerate bitsandbytes pandas requests

# Verify
horovodrun --check-build

# --- 5. Passwordless SSH (run only on the MASTER VM, pointing at each worker) ---
# ssh-keygen -t rsa -b 4096
# ssh-copy-id user@worker-vm-ip

echo "Setup complete. Create a hostfile on the master VM before running training."
