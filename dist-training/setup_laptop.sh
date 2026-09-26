#!/bin/bash
# ============================================================
# LAPTOP SETUP — run once. Requires WSL2 Ubuntu if on Windows.
# Uses Horovod's Gloo backend — no MPI, no cluster needed.
# ============================================================

pip install torch
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch] --no-cache-dir
pip install peft transformers datasets accelerate bitsandbytes pandas requests

# Verify the install
horovodrun --check-build
