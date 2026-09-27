
sudo apt update
sudo apt install -y software-properties-common
sudo apt update
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1 nvidia-driver
echo ">>> REBOOT NOW, then continue with the rest of this script <<<"

nvidia-smi


sudo apt install -y openmpi-bin libopenmpi-dev

pip install torch --index-url https://download.pytorch.org/whl/cu121
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch] --no-cache-dir
pip install peft transformers datasets accelerate bitsandbytes pandas requests petastorm pyarrow

horovodrun --check-build


echo "Setup complete. Create a hostfile on the master VM before running training."
