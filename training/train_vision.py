import os
import torch
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
    Resize,
    ToTensor
)
import evaluate
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

def compute_metrics(eval_pred):
    """
    Compute metrics: Accuracy, F1, Precision, Recall.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }

def train_vision_model(
    data_dir="dataset/images",
    model_name="google/vit-base-patch16-224",
    output_dir="models/finetuned_vit_fahad",
    num_epochs=3,
    batch_size=8
):
    print(f"Loading dataset from {data_dir}...")
    # Load dataset from folder structure
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    
    # Split into train and validation
    # If there is no validation split, create one
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        dataset["validation"] = dataset["test"]
        del dataset["test"]
        
    labels = dataset["train"].features["label"].names
    print(f"Found labels: {labels}")
    
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    
    print(f"Loading processor for {model_name}...")
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Define transforms
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = (
        processor.size["shortest_edge"]
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    
    _train_transforms = Compose([
        Resize(size),
        RandomHorizontalFlip(),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ToTensor(),
        normalize,
    ])
    
    _val_transforms = Compose([
        Resize(size),
        ToTensor(),
        normalize,
    ])
    
    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples
        
    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples
    
    print("Applying transforms...")
    # Set transforms
    dataset["train"].set_transform(train_transforms)
    dataset["validation"].set_transform(val_transforms)
    
    print(f"Loading model {model_name}...")
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print("Evaluating...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")
    
    # Detailed Evaluation
    print("\nGenerating Detailed Classification Report...")
    predictions = trainer.predict(dataset["validation"])
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=labels))

if __name__ == "__main__":
    train_vision_model()
