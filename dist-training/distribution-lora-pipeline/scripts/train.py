import numpy as np
import torch
import horovod.torch as hvd
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader
from petastorm.transform import TransformSpec
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

#Horovod setup 
hvd.init()
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"  # swap for your real base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["c_attn"],  # model-specific — check print(model) for your real base
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
if hvd.rank() == 0:
    model.print_trainable_parameters()

MAX_LENGTH = 512

.
def tokenize_batch(batch_df):
    input_ids_col, attention_mask_col, labels_col = [], [], []
    for context, target in zip(batch_df["context"], batch_df["target"]):
        context_ids = tokenizer(str(context), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(str(target), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        ids = (context_ids + target_ids)[:MAX_LENGTH]
        labels = ([-100] * len(context_ids) + target_ids)[:MAX_LENGTH]

        pad_len = MAX_LENGTH - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        input_ids_col.append(np.array(ids, dtype=np.int64))
        attention_mask_col.append(np.array(attention_mask, dtype=np.int64))
        labels_col.append(np.array(labels, dtype=np.int64))

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

# ── 5. Petastorm reads Spark's Parquet directly, sharded per worker ─
# Path can be local ("file:///...") or the same S3/HDFS path Spark wrote to.
PARQUET_PATH = "file:///path/to/cleaned_dialect_news.parquet"  # <-- fill in, same on every machine
BATCH_SIZE = 8
EPOCHS = 3


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0

    
    with make_batch_reader(
        PARQUET_PATH,
        cur_shard=hvd.rank(),
        shard_count=hvd.size(),
        num_epochs=1,
        transform_spec=transform_spec,
    ) as reader:
        loader = PetastormDataLoader(reader, batch_size=BATCH_SIZE)
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if hvd.rank() == 0 and step % 10 == 0:
                print(f"Epoch {epoch} step {step} loss {loss.item():.4f}")

    if hvd.rank() == 0 and num_batches > 0:
        print(f"Epoch {epoch} avg loss: {total_loss / num_batches:.4f}")

if hvd.rank() == 0:
    model.save_pretrained("./lora_adapter_output")
    print("Adapter saved.")
