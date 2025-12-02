import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np
from text_processing import preprocess_text



def compute_metrics(eval_pred):
    """
    Compute metrics: Accuracy, F1, Precision.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"]
    }

def finetune_sentiment(
    dataset_path="dataset/product_review_dataset.csv",
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
    output_dir="models/finetuned_roberta",
    num_epochs=1,
    max_samples=200
):
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Check columns
    if 'reviews' not in df.columns or 'sentiment' not in df.columns:
        print(f"Error: Dataset must contain 'reviews' and 'sentiment' columns. Found: {df.columns}")
    

    print("Preprocessing text...")
    df['processed_text'] = df['reviews'].apply(preprocess_text)
    
    # Map sentiment to labels
    # Check unique values in sentiment
    print(f"Unique sentiments: {df['sentiment'].unique()}")
    
    # Standardize sentiment labels if needed
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    # Handle potential casing issues or extra whitespace
    df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
    
    # Filter out unknown sentiments
    df = df[df['sentiment'].isin(label_map.keys())]
    
    df['label'] = df['sentiment'].map(label_map)
    
    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["processed_text"], truncation=True, padding=True, max_length=512)
    
    print("Tokenizing dataset...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print(f"Loading model {model_name}...")
    # use_safetensors=True to avoid torch vulnerability warning
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, use_safetensors=True)
    
    # LoRA Configuration
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=16, # Increased r for better adaptation
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["query", "value"] # Target attention modules for RoBERTa
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4, # Slightly higher LR for LoRA
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        metric_for_best_model="f1"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on validation set
    print("Evaluating...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    finetune_sentiment()
