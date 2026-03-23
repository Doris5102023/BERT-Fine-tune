import os
import time
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict

# ===================== 【Modify this to your local path】 =====================
DATA_DIR = "./data/SST2"  # Path to your local SST-2 dataset folder
# =====================================================================

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Fix SST-2 dataset format (resolve column mismatch issue)
def fix_sst2_format(file_path):
    """
    Fix tsv file format: force convert to two columns (sentence, label) and handle messy column names
    """
    # Read file (without specifying column names, read raw content first)
    df = pd.read_csv(file_path, sep="\t", header=None, on_bad_lines="skip")
    
    # Ensure only two columns (SST-2 standard format)
    if df.shape[1] > 2:
        df = df.iloc[:, :2]  # Keep only first two columns
    elif df.shape[1] < 2:
        raise ValueError(f"File {file_path} has less than 2 columns, please check data integrity")
    
    # Rename columns (force to sentence and label)
    df.columns = ["sentence", "label"]
    
    # Clean data: convert label to integer and fill missing values
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["sentence"] = df["sentence"].astype(str).str.strip()
    
    # Filter empty text
    df = df[df["sentence"] != ""].reset_index(drop=True)
    return df

# Load and fix local SST-2 dataset
def load_and_fix_sst2(data_dir):
    data_splits = {}
    # Iterate through train/dev/test files
    for split in ["train", "dev", "test"]:
        file_path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Fix format
        df = fix_sst2_format(file_path)
        # Convert to Hugging Face Dataset format
        data_splits[split] = Dataset.from_pandas(df)
    
    # Build DatasetDict (compatible with Trainer)
    dataset = DatasetDict({
        "train": data_splits["train"],
        "validation": data_splits["dev"],  # dev corresponds to validation
        "test": data_splits["test"]
    })
    return dataset

# Data preprocessing
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

# Compute accuracy + F1 score manually (pure NumPy, no third-party library dependency)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Calculate predictions manually (argmax)
    predictions = np.argmax(logits, axis=-1)
    
    # 1. Calculate accuracy
    correct = np.sum(predictions == labels)
    total = len(labels)
    accuracy = round(correct / total, 4)
    
    # 2. Calculate F1 score manually (binary classification, macro-average)
    # Step 1: Calculate TP/TN/FP/FN
    tp = np.sum((predictions == 1) & (labels == 1))  # True Positive
    tn = np.sum((predictions == 0) & (labels == 0))  # True Negative
    fp = np.sum((predictions == 1) & (labels == 0))  # False Positive
    fn = np.sum((predictions == 0) & (labels == 1))  # False Negative
    
    # Step 2: Calculate Precision and Recall
    # Avoid division by zero (add small value)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # Step 3: Calculate F1 score (F1 = 2*(P*R)/(P+R))
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1 = round(f1, 4)
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }

# Fine-tune single model
def finetune(model_name, dataset, device):
    print(f"\n========== Finetuning {model_name} ==========")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # Tokenize data
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    # Training hyperparameters
    LR = 2e-5
    BATCH_SIZE = 16
    EPOCHS = 3

    # TrainingArguments compatible with old/new versions
    training_args = TrainingArguments(
        output_dir=f"./output/{model_name}",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",  # Parameter for old versions (replaces evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=42,
        logging_steps=50,
        report_to="none",  # Disable wandb to avoid extra dependencies
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,  # Use manually implemented evaluation function
    )

    # Training timing
    start = time.time()
    trainer.train()
    train_time = time.time() - start

    # Evaluate on test set
    test_result = trainer.evaluate(tokenized_datasets["test"])

    # Result summary
    result = {
        "model": model_name,
        "test_acc": test_result["eval_accuracy"],
        "test_f1": test_result["eval_f1"],  # Add F1 result
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": 42,
        "train_time_sec": round(train_time, 2),
        "train_time_min": round(train_time / 60, 2),
        "device": str(device),
    }
    return result

# ===================== Main Program =====================
if __name__ == "__main__":
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    # Load local data (with format fix)
    try:
        dataset = load_and_fix_sst2(DATA_DIR)
        print(f"Dataset loaded successfully!")
        print(f"Training set samples: {len(dataset['train'])}")
        print(f"Validation set samples: {len(dataset['validation'])}")
        print(f"Test set samples: {len(dataset['test'])}")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        exit(1)

    # Select models to fine-tune (replaceable: albert-base-v2 / roberta-base)
    model_list = ["bert-base-uncased", "distilbert-base-uncased"]

    # Fine-tune models sequentially
    results = []
    for model in model_list:
        res = finetune(model, dataset, device)
        results.append(res)

    # Print result table
    df = pd.DataFrame(results)
    print("\n==================== Final Results ====================")
    print(df.to_string(index=False))

    # Save results to CSV (for report writing)
    df.to_csv("SST2_finetune_results.csv", index=False)
    print("\nResults saved to SST2_finetune_results.csv")

    # Model comparison analysis
    print("\n==================== Model Comparison ====================")
    bert = results[0]
    distil = results[1]

    print(f"BERT  Accuracy: {bert['test_acc']:.4f} | F1: {bert['test_f1']:.4f} | Time: {bert['train_time_min']} min")
    print(f"DistilBERT Accuracy: {distil['test_acc']:.4f} | F1: {distil['test_f1']:.4f} | Time: {distil['train_time_min']} min")

    if bert["test_acc"] > distil["test_acc"]:
        print("\nConclusion: BERT-base has higher accuracy but is slower to train and has larger parameter size.")
    else:
        print("\nConclusion: DistilBERT has comparable accuracy but is faster to train and more lightweight.")
