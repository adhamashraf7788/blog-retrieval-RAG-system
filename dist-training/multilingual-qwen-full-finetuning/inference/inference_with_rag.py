"""
Reference integration for the RAG service.

This loads the complete model produced by scripts/train.py.
No PEFT or LoRA adapter is required.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.getenv("MODEL_PATH", "./full_model_output")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

def generate_from_context(retrieved_context: str, max_new_tokens: int = 120) -> str:
    if not retrieved_context or not retrieved_context.strip():
        raise ValueError("retrieved_context must be non-empty")

    prompt = (
        "Summarize the following retrieved news context in 2-3 concise "
        "sentences. Keep the same language as the source. For Arabic, use "
        "Modern Standard Arabic. Do not invent facts.\n\n"
        f"Context:\n{retrieved_context}\n\nSummary:"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
