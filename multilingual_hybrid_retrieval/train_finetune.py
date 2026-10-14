import json
import random
from pathlib import Path

import torch
from datasets import Dataset

from peft import LoraConfig, TaskType

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.sentence_transformer.training_args import BatchSamplers


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path(
    "/content/drive/MyDrive/rag-arabic-english/expermet_finetune_embedding"
)

TRAIN_PATH = BASE_DIR / "data" / "train_scored_sample.jsonl"

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "bge-m3-egyptian-ar-en-lora-v2-colab"
)

MODEL_NAME = "BAAI/bge-m3"

MODEL_REVISION = "142964af7e05de16511657561de8e8750fc153a0"


# ============================================================
# 2. Training configuration
# ============================================================

MAX_TRAIN_SAMPLES = 20_000

EPOCHS = 1

PER_DEVICE_BATCH_SIZE = 4

MINI_BATCH_SIZE = 1

LEARNING_RATE = 2e-5

MAX_SEQ_LENGTH = 96

WARMUP_RATIO = 0.1

RANDOM_SEED = 42


# ============================================================
# 3. Reproducibility
# ============================================================

random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 4. Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 60)
print("DEVICE")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    total_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(
        "GPU Memory:",
        round(total_memory, 2),
        "GB"
    )


# ============================================================
# 5. Check paths
# ============================================================

print("\n" + "=" * 60)
print("PATH CHECK")
print("=" * 60)

print("BASE_DIR:")
print(BASE_DIR)

print("\nTRAIN_PATH:")
print(TRAIN_PATH)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)


if not TRAIN_PATH.exists():

    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_PATH}"
    )


print("\nTraining file exists.")


# ============================================================
# 6. Load dataset
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

records = []


with open(
    TRAIN_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        row = json.loads(line)

        arabic = str(
            row["input"]
        ).strip()

        english = str(
            row["output"]
        ).strip()

        if arabic and english:

            records.append(
                {
                    "anchor": arabic,
                    "positive": english,
                }
            )


print(
    "Available pairs:",
    len(records)
)


# ============================================================
# 7. Shuffle and select subset
# ============================================================

random.shuffle(records)

records = records[:MAX_TRAIN_SAMPLES]


print(
    "Training pairs:",
    len(records)
)


if len(records) < MAX_TRAIN_SAMPLES:

    print(
        "\nWARNING:"
        "\nThe dataset contains fewer pairs than "
        "MAX_TRAIN_SAMPLES."
    )


# ============================================================
# 8. Convert to Hugging Face Dataset
# ============================================================

train_dataset = Dataset.from_dict(
    {
        "anchor": [
            row["anchor"]
            for row in records
        ],

        "positive": [
            row["positive"]
            for row in records
        ],
    }
)


print(
    "Dataset created."
)

print(
    "Dataset size:",
    len(train_dataset)
)


# ============================================================
# 9. Load BGE-M3
# ============================================================

print("\n" + "=" * 60)
print("LOADING BGE-M3")
print("=" * 60)


model = SentenceTransformer(

    MODEL_NAME,

    revision=MODEL_REVISION,

    device=device,

    model_kwargs={
        "use_safetensors": True,
    },
)


model.max_seq_length = MAX_SEQ_LENGTH


print(
    "Model loaded."
)

print(
    "Max sequence length:",
    model.max_seq_length
)


# ============================================================
# 10. Add LoRA
# ============================================================

print("\n" + "=" * 60)
print("ADDING LoRA")
print("=" * 60)


lora_config = LoraConfig(

    task_type=TaskType.FEATURE_EXTRACTION,

    inference_mode=False,

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "query",
        "value",
    ],
)


model.add_adapter(
    lora_config
)


print(
    "LoRA adapter added."
)


# ============================================================
# 11. Trainable parameters
# ============================================================

print("\n" + "=" * 60)
print("TRAINABLE PARAMETERS")
print("=" * 60)


trainable_params = 0

all_params = 0


for param in model.parameters():

    all_params += param.numel()

    if param.requires_grad:

        trainable_params += param.numel()


print(
    "Trainable parameters:",
    trainable_params
)

print(
    "All parameters:",
    all_params
)

print(
    "Trainable percentage:",
    round(
        100 * trainable_params / all_params,
        4
    ),
    "%"
)


# ============================================================
# 12. Cached Multiple Negatives Ranking Loss
# ============================================================

print("\n" + "=" * 60)
print("LOSS")
print("=" * 60)


train_loss = losses.CachedMultipleNegativesRankingLoss(

    model=model,

    mini_batch_size=MINI_BATCH_SIZE,

    show_progress_bar=True,
)


print(
    "Using CachedMultipleNegativesRankingLoss"
)

print(
    "Effective batch size:",
    PER_DEVICE_BATCH_SIZE
)

print(
    "Mini batch size:",
    MINI_BATCH_SIZE
)


# ============================================================
# 13. Training configuration
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CONFIG")
print("=" * 60)


steps_per_epoch = (
    len(train_dataset)
    // PER_DEVICE_BATCH_SIZE
)


total_steps = (
    steps_per_epoch
    * EPOCHS
)


warmup_steps = max(
    1,
    int(
        total_steps
        * WARMUP_RATIO
    )
)


print(
    "Samples:",
    len(train_dataset)
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    PER_DEVICE_BATCH_SIZE
)

print(
    "Mini batch:",
    MINI_BATCH_SIZE
)

print(
    "Steps:",
    total_steps
)

print(
    "Warmup steps:",
    warmup_steps
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Max sequence length:",
    MAX_SEQ_LENGTH
)

print(
    "FP16:",
    True
)


# ============================================================
# 14. Training arguments
# ============================================================

training_args = SentenceTransformerTrainingArguments(

    output_dir=str(
        OUTPUT_DIR
    ),

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    warmup_steps=warmup_steps,

    fp16=True,

    bf16=False,

    batch_sampler=BatchSamplers.NO_DUPLICATES,

    logging_steps=50,

    save_strategy="steps",

    save_steps=500,

    save_total_limit=2,

    report_to="none",

    dataloader_num_workers=0,

    seed=RANDOM_SEED,
)


# ============================================================
# 15. Trainer
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAINER")
print("=" * 60)


trainer = SentenceTransformerTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    loss=train_loss,
)


print(
    "Trainer created."
)


# ============================================================
# 16. Train
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

print(
    f"Training {len(train_dataset):,} pairs "
    f"for {EPOCHS} epoch(s)."
)

print(
    f"Expected training steps: {total_steps:,}"
)


trainer.train()


# ============================================================
# 17. Save model
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)


model.save_pretrained(
    str(OUTPUT_DIR)
)


print(
    "Model saved to:"
)

print(
    OUTPUT_DIR
)


# ============================================================
# 18. Final message
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)


print(
    "Fine-tuning completed successfully."
)

print(
    "Training pairs:",
    len(train_dataset)
)

print(
    "Output directory:",
    OUTPUT_DIR
)