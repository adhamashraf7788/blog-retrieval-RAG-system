"""
This is NOT something you run yourself — it's a reference for whoever
owns the RAG service's code, showing what changes in THEIR model-loading
step once your adapter is ready.

Hand them the lora_adapter_output/ folder (a few MB) via shared storage,
S3, or however your team already moves files between services, then this
is the change on their end.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "gpt2"                          # must match what train.py used
ADAPTER_PATH = "/shared/path/lora_adapter_output"  # wherever the adapter file lands on their service

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

# The only real change: wrap the base model with the trained adapter.
# Everything about how the RAG service retrieves documents and builds
# prompts stays exactly as it already is.
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Wherever their service currently calls the base model to generate text,
# it now calls this `model` object instead — same .generate() interface.
