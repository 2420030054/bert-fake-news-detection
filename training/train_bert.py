
import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Select the available device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

# 2. Load the existing train/test datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 3. Map the three class names to numeric IDs
label2id = {
    "false-negative": 0,
    "other-neutral": 1,
    "true-positive": 2,
}
id2label = {value: key for key, value in label2id.items()}

train_df["label"] = train_df["label"].map(label2id)
test_df["label"] = test_df["label"].map(label2id)

if train_df["label"].isnull().any() or test_df["label"].isnull().any():
    raise ValueError("Unknown label found in the CSV files.")

train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

# 4. Load the pretrained BERT tokenizer
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 5. Convert the CSV data into Hugging Face datasets
train_dataset = Dataset.from_pandas(
    train_df[["tweet", "label"]], preserve_index=False
)
test_dataset = Dataset.from_pandas(
    test_df[["tweet", "label"]], preserve_index=False
)

# 6. Convert tweet text into tokens BERT understands
def tokenize_batch(batch):
    return tokenizer(
        batch["tweet"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

# Trainer expects the target column to be named "labels"
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

columns = ["input_ids", "attention_mask", "labels"]
train_dataset.set_format(type="torch", columns=columns)
test_dataset.set_format(type="torch", columns=columns)

# 7. Load BERT with a three-class classification head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)

# 8. Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

# 9. Configure training
training_args = TrainingArguments(
    output_dir="models/bert_checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
    use_cpu=(device == "cpu"),
    dataloader_pin_memory=False,
)

# 10. Train and evaluate
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting BERT fine-tuning...")
trainer.train()

print("\nEvaluating BERT...")
results = trainer.evaluate()
print(results)

# 11. Save the final model and tokenizer
save_dir = "models/bert_fake_news"
os.makedirs(save_dir, exist_ok=True)

trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print("\nBERT model and tokenizer saved to:", save_dir)
