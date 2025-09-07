"""
SDG Multi-label Regression Classifier for Patent Main Text

This script trains a transformer-based regression model (e.g., bert-for.-patents) to predict SDG label distributions from patent main text.
It uses the output of the semantic_match step (the training split, with main_text and SDG vectors) as input.

Requirements:
- Python packages: pandas, numpy, torch, transformers, datasets, scikit-learn, skmultilearn
- Input: CSV file (from semantic_match, e.g., data_train.csv) with columns:
    - main_text: The patent text to use as input
    - sdg_vector_norm: Normalized SDG label vector (target for regression)
    - sdg_vector: Raw SDG label counts (for class balancing)
    - SDG_vector: Binary SDG label vector (for stratified splitting)
- Output: Trained model and evaluation metrics

Example usage:
    python classifier.py
    # or see the get_aligned_train_val_split usage at the bottom

This script is designed to be run after the data_alignment and semantic_match steps.
"""
import ast
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from skmultilearn.model_selection import iterative_train_test_split

# ----------------------
# Device setup
# ----------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ----------------------
# Utility: Stratified multi-label split (aligned with semantic_match.py)
# ----------------------
def get_aligned_train_val_split(aligned_df, test_size=0.2, random_seed=42, sdg_vector_col='SDG_vector'):
    """
    Stratified multi-label split for the aligned dataset, using the SDG_vector column (binary vector per patent).
    Returns train and validation DataFrames, aligned for downstream use.
    This matches the logic used in semantic_match.py and data_alignment.py.
    """
    X_matrix = aligned_df.index.values.reshape(-1, 1)
    y_matrix = np.vstack(aligned_df[sdg_vector_col].apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array(ast.literal_eval(x))))
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_matrix, y_matrix, test_size=test_size
    )
    train_df = aligned_df[aligned_df.index.isin(X_train.reshape(-1,).tolist())].reset_index(drop=True)
    val_df = aligned_df[aligned_df.index.isin(X_val.reshape(-1,).tolist())].reset_index(drop=True)
    return train_df, val_df

# ----------------------
# File paths (edit as needed)
# ----------------------
train_csv_path = "../vectors/data_train.csv"  # Output of semantic_match/data_alignment
output_dir = "./best_sdg_transformer"

# ----------------------
# Data loading and preprocessing
# ----------------------
print("Loading training data...")
silver_df = pd.read_csv(train_csv_path)
silver_df["sdg_vector_norm"] = silver_df.sdg_vector_norm.apply(ast.literal_eval)
silver_df["sdg_vector"] = silver_df.sdg_vector.apply(ast.literal_eval)
silver_df["SDG_vector_binary"] = silver_df.SDG_vector.apply(lambda s: [0 if c==0 else 1 for c in ast.literal_eval(s)])

# Use the aligned split function for train/val
train_df, val_df = get_aligned_train_val_split(silver_df, test_size=0.2, sdg_vector_col='SDG_vector_binary')

# Compute class weights for SDG regression (inverse frequency)
all_sdg_counts = np.sum(np.vstack(train_df['sdg_vector'].values), axis=0)
class_freq = all_sdg_counts / np.sum(all_sdg_counts)
class_weights = 1.0 / (class_freq + 1e-6)
class_weights = class_weights / np.sum(class_weights)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# ----------------------
# Convert to HuggingFace datasets
# ----------------------
train_dataset = Dataset.from_pandas(train_df[['main_text', 'sdg_vector_norm']].rename(columns={'sdg_vector_norm': 'labels'}))
val_dataset = Dataset.from_pandas(val_df[['main_text', 'sdg_vector_norm']].rename(columns={'sdg_vector_norm': 'labels'}))

# ----------------------
# Model and tokenizer setup
# ----------------------
model_name = 'anferico/bert-for-patents'  # Change to your preferred model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=17,  # One regression output per SDG
    problem_type='regression'
)

# ----------------------
# Custom Trainer for weighted MSE loss
# ----------------------
class WeightedMSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.MSELoss(reduction='none')
        loss_per_dim = loss_fct(logits, labels)
        weighted_loss = (loss_per_dim * class_weights_tensor.to(loss_per_dim.device)).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss

# ----------------------
# Tokenize dataset
# ----------------------
def tokenize_function(examples):
    return tokenizer(examples['main_text'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# ----------------------
# Metrics
# ----------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    r2 = r2_score(labels, predictions)
    per_class_mse = np.mean((predictions - labels) ** 2, axis=0)
    per_class_mse_dict = {f"mse_sdg_{i+1}": per_class_mse[i] for i in range(len(per_class_mse))}
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }
    metrics.update(per_class_mse_dict)
    return metrics

# ----------------------
# Training arguments
# ----------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,
    logging_dir="./logs",
    logging_strategy="epoch",
    report_to="none"
)

# ----------------------
# Run training
# ----------------------
trainer = WeightedMSETrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nTraining complete. Best model and logs saved to:", output_dir)

# Example usage for splitting:
# from classifier import get_aligned_train_val_split
# df = pd.read_csv('vectors/data_train.csv')
# df['SDG_vector'] = df['SDG_vector'].apply(ast.literal_eval)
# train_df, val_df = get_aligned_train_val_split(df)
