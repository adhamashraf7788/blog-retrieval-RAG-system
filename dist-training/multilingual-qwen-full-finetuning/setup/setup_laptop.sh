#!/bin/bash
# ============================================================
# LAPTOP SETUP — run once. Requires WSL2 Ubuntu if on Windows.
# Uses Horovod's Gloo backend — no MPI, no cluster needed.
# CPU-only build, matching the production VMs (no GPUs there).
# ============================================================

pip install torch --index-url https://download.pytorch.org/whl/cpu
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch] --no-cache-dir
pip install peft transformers datasets accelerate pandas requests petastorm pyarrow

# Verify the install
horovodrun --check-build
