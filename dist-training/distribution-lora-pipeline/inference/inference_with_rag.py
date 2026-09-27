

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "gpt2"                          # must match what train.py used
ADAPTER_PATH = "/shared/path/lora_adapter_output"  # wherever the adapter file lands 

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)


model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


