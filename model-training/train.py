import torch
import pandas as pd
import horovod.torch as hvd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ── 1. Horovod setup ──────────────────────────────────────────────
hvd.init()
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. Load frozen base model + tokenizer ─────────────────────────
model_name = "gpt2"  # swap for your real base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# ── 3. Attach LoRA adapter ─────────────────────────────────────────
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["c_attn"],  # model-specific — check print(model) for your real base
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if hvd.rank() == 0:
    model.print_trainable_parameters()

# ── 4. Dataset class wrapping your CSV ─────────────────────────────
class DialectNewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, text_column="text", max_length=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx][self.text_column])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # causal LM: predict the same sequence, shifted internally
        }

# ── 5. Data loading — each worker sees a different slice ──────────
csv_path = "cleaned_dialect_news.csv"  # path on each machine — same filename, per-region data
dataset = DialectNewsDataset(csv_path, tokenizer)

sampler = DistributedSampler(
    dataset,
    num_replicas=hvd.size(),
    rank=hvd.rank(),
    shuffle=True,
)
loader = DataLoader(dataset, batch_size=8, sampler=sampler)

# ── 6. Optimizer wrapped by Horovod ────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# ── 7. Training loop ────────────────────────────────────────────────
model.train()
epochs = 3
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # reshuffles differently each epoch — important for DistributedSampler
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if hvd.rank() == 0 and step % 10 == 0:
            print(f"Epoch {epoch} step {step} loss {loss.item():.4f}")

    if hvd.rank() == 0:
        print(f"Epoch {epoch} avg loss: {total_loss / len(loader):.4f}")

# ── 8. Save adapter (rank 0 only) ─────────────────────────────────
if hvd.rank() == 0:
    model.save_pretrained("./lora_adapter_output")
    print("Adapter saved.")
