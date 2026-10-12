"""
Distributed FULL fine-tuning (no LoRA / no PEFT) for Arabic, English, and French.

Laptop smoke test:
  horovodrun -np 2 --gloo python scripts/train.py

Multi-VM CPU cluster:
  horovodrun -np 2 -hostfile config/hostfile python scripts/train.py

Configure DATA_PATH, MODEL_NAME, and output path through environment variables.
Every worker must be able to read the same Parquet dataset and write access
is needed only on rank 0 for the final model output.
"""

import os
import numpy as np
import torch
import horovod.torch as hvd
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader
from petastorm.transform import TransformSpec
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Distributed setup
hvd.init()
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")  # smoke-test default; not multilingual-specialized
DATA_PATH = os.getenv(
    "DATA_PATH", "file:///path/to/cleaned_dialect_news.parquet"
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./full_model_output")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))

# 2. Load the same complete base model on every worker.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
# Full fine-tuning: all model parameters are trainable.
for parameter in model.parameters():
    parameter.requires_grad = True

if hvd.rank() == 0:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME}")
    print(f"Total parameters: {total:,}; trainable parameters: {trainable:,}")
    print(f"Horovod workers: {hvd.size()}; device: {device}")

def tokenize_batch(batch_df):
    input_ids_col, attention_mask_col, labels_col = [], [], []
    for context, target in zip(batch_df["context"], batch_df["target"]):
        context_ids = tokenizer(
            str(context), add_special_tokens=False
        )["input_ids"]
        target_ids = tokenizer(
            str(target), add_special_tokens=False
        )["input_ids"] + [tokenizer.eos_token_id]

        # Preserve the end of the context if the sequence is too long.
        max_context = max(0, MAX_LENGTH - min(len(target_ids), MAX_LENGTH))
        context_ids = context_ids[-max_context:] if max_context else []
        target_ids = target_ids[:MAX_LENGTH]
        ids = (context_ids + target_ids)[:MAX_LENGTH]
        labels = ([-100] * len(context_ids) + target_ids)[:MAX_LENGTH]

        pad_len = MAX_LENGTH - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len

        input_ids_col.append(np.asarray(ids, dtype=np.int64))
        attention_mask_col.append(np.asarray(attention_mask, dtype=np.int64))
        labels_col.append(np.asarray(labels, dtype=np.int64))

    batch_df["input_ids"] = input_ids_col
    batch_df["attention_mask"] = attention_mask_col
    batch_df["labels"] = labels_col
    return batch_df[["input_ids", "attention_mask", "labels"]]

transform_spec = TransformSpec(
    tokenize_batch,
    edit_fields=[
        ("input_ids", np.int64, (MAX_LENGTH,), False),
        ("attention_mask", np.int64, (MAX_LENGTH,), False),
        ("labels", np.int64, (MAX_LENGTH,), False),
    ],
    removed_fields=["context", "target"],
)

# 3. Every worker starts with identical model parameters.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters()
)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# 4. Petastorm shards records so workers process different data.
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    with make_batch_reader(
        DATA_PATH,
        cur_shard=hvd.rank(),
        shard_count=hvd.size(),
        num_epochs=1,
        transform_spec=transform_spec,
    ) as reader:
        loader = PetastormDataLoader(reader, batch_size=BATCH_SIZE)
        for step, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            # Horovod's distributed optimizer averages gradients across workers.
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            num_batches += 1
            if hvd.rank() == 0 and step % 10 == 0:
                print(f"epoch={epoch + 1} step={step} loss={loss.item():.4f}")

    # Aggregate metrics for reporting across the whole cluster.
    local_stats = torch.tensor(
        [total_loss, num_batches], dtype=torch.float64, device=device
    )
    global_stats = hvd.allreduce(local_stats, op=hvd.Sum)
    if hvd.rank() == 0:
        if global_stats[1].item() > 0:
            avg_loss = global_stats[0].item() / global_stats[1].item()
            print(f"Epoch {epoch + 1} global average loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}: no batches were processed.")

# 5. Save the complete fine-tuned model, not an adapter.
if hvd.rank() == 0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Full fine-tuned model and tokenizer saved to: {OUTPUT_DIR}")
