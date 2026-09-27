pip install torch
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch] --no-cache-dir
pip install peft transformers datasets accelerate bitsandbytes pandas requests petastorm pyarrow

# Verify the install
horovodrun --check-build
