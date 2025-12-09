
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional
from fusion_mlp import MultimodalFusionWithAttention
from transformers import (
    AutoTokenizer,
    RobertaModel,
    ViTImageProcessor,
    ViTModel
)


# ==============================
# CONFIG
# ==============================
class Config:
    # Training hyperparameters
    BATCH_SIZE = 16  # Reduced due to sequence processing memory
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 25
    DROPOUT = 0.2
    ATTENTION_DROPOUT = 0.1
    NUM_HEADS = 8

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model paths - adjust these as needed
    VIT_PATH = "/content/drive/MyDrive/trained/finetuned_vit_fahad"
    ROBERTA_PATH = "/content/drive/MyDrive/trained/finetuned_roberta_fahad"

    # Data paths - adjust these as needed
    DATA_PATH = "/content/drive/MyDrive/CrossAttention/products_augmented.csv"
    IMAGE_DIR = "/content/drive/MyDrive/CrossAttention/"

    # Save path
    SAVE_PATH = "/content/drive/MyDrive/trained/CrossAttentMLP.pth"

    # Text processing
    MAX_TEXT_LENGTH = 128

    # Embedding dimension (ViT-Base and RoBERTa-Base both use 768)
    EMBED_DIM = 768


config = Config()


# ==============================
# FEATURE EXTRACTORS (SEQUENCE MODE)
# ==============================
class SequenceFeatureExtractors:


    def __init__(self, vit_path: str, roberta_path: str, device: torch.device):
        print("\nLoading pretrained models for sequence extraction...")

        # ViT Model
        self.vit_model = ViTModel.from_pretrained(vit_path).to(device).eval()
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_path)

        # RoBERTa Model
        self.roberta_model = RobertaModel.from_pretrained(roberta_path).to(device).eval()
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path)

        self.device = device
        print("Models loaded successfully!")

    @torch.no_grad()
    def get_image_sequence(self, pixel_values: torch.Tensor) -> torch.Tensor:

        pixel_values = pixel_values.to(self.device)
        outputs = self.vit_model(pixel_values, output_hidden_states=False)
        # last_hidden_state contains the full sequence
        return outputs.last_hidden_state  # (B, N_patches+1, D)

    @torch.no_grad()
    def get_text_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.roberta_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        return outputs.last_hidden_state  # (B, seq_len, D)


# ==============================
# DATASET (RAW INPUTS)
# ==============================
class FusionDatasetRaw(Dataset):


    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        vit_processor: ViTImageProcessor,
        tokenizer: AutoTokenizer,
        max_length: int = 128
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.vit_processor = vit_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Load and process image
        img_path = os.path.join(self.image_dir, row['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
            pixel_values = self.vit_processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].squeeze(0)  # (3, 224, 224)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Return zeros for failed images
            pixel_values = torch.zeros(3, 224, 224)

        # Process text
        text = str(row['review_text']) if pd.notna(row['review_text']) else "No review"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized['input_ids'].squeeze(0)  # (seq_len,)
        attention_mask = tokenized['attention_mask'].squeeze(0)  # (seq_len,)

        # Normalize score to [0, 1]
        # Assuming score is in range 1-10
        score = float(row['score']) if pd.notna(row['score']) else 5.0  # Default to middle score if NaN
        normalized_score = (score - 1) / 9.0

        # Clamp to valid range to avoid NaN issues
        normalized_score = max(0.0, min(1.0, normalized_score))

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'score': torch.tensor(normalized_score, dtype=torch.float32),
            'original_score': torch.tensor(score, dtype=torch.float32)
        }


# ==============================
# TRAINING FUNCTIONS
# ==============================
def train_epoch(
    model: nn.Module,
    feature_extractors: SequenceFeatureExtractors,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        # Extract sequence features (frozen backbones)
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['score'].to(device)

        # Get sequence embeddings from frozen backbones
        with torch.no_grad():
            image_seq = feature_extractors.get_image_sequence(pixel_values)
            text_seq = feature_extractors.get_text_sequence(input_ids, attention_mask)

        # Forward pass through fusion model (trainable)
        optimizer.zero_grad()
        pred = model(
            image_seq=image_seq,
            text_seq=text_seq,
            text_attention_mask=attention_mask
        )

        # Compute loss
        loss = loss_fn(pred, target)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    feature_extractors: SequenceFeatureExtractors,
    loader: DataLoader,
    device: torch.device
) -> tuple:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            original_scores = batch['original_score'].numpy()

            # Get sequence embeddings
            image_seq = feature_extractors.get_image_sequence(pixel_values)
            text_seq = feature_extractors.get_text_sequence(input_ids, attention_mask)

            # Forward pass
            pred = model(
                image_seq=image_seq,
                text_seq=text_seq,
                text_attention_mask=attention_mask
            )

            # Convert predictions back to 1-10 scale
            pred_original = 1 + pred.cpu().numpy() * 9

            all_preds.extend(pred_original)
            all_targets.extend(original_scores)

    # Convert to numpy arrays and filter out NaN values
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Check for NaN and filter them out
    valid_mask = ~(np.isnan(all_preds) | np.isnan(all_targets))
    if not np.all(valid_mask):
        print(f"Warning: Found {np.sum(~valid_mask)} NaN values, filtering them out")
        all_preds = all_preds[valid_mask]
        all_targets = all_targets[valid_mask]

    if len(all_preds) == 0:
        print("Error: All predictions are NaN!")
        return float('inf'), float('inf'), 0.0

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return rmse, mae, r2


# ==============================
# MAIN TRAINER
# ==============================
def train_fusion_model():
    print(f"Using device: {config.DEVICE}")

    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(config.DATA_PATH)
    print(f"Total samples: {len(df)}")

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Initialize feature extractors
    feature_extractors = SequenceFeatureExtractors(
        vit_path=config.VIT_PATH,
        roberta_path=config.ROBERTA_PATH,
        device=config.DEVICE
    )

    # Create datasets
    train_dataset = FusionDatasetRaw(
        df=train_df,
        image_dir=config.IMAGE_DIR,
        vit_processor=feature_extractors.vit_processor,
        tokenizer=feature_extractors.roberta_tokenizer,
        max_length=config.MAX_TEXT_LENGTH
    )
    val_dataset = FusionDatasetRaw(
        df=val_df,
        image_dir=config.IMAGE_DIR,
        vit_processor=feature_extractors.vit_processor,
        tokenizer=feature_extractors.roberta_tokenizer,
        max_length=config.MAX_TEXT_LENGTH
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    # Initialize fusion model
    model = MultimodalFusionWithAttention(
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        mlp_hidden_dims=(512, 128, 32),
        dropout=config.DROPOUT,
        attention_dropout=config.ATTENTION_DROPOUT
    ).to(config.DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=3,
    #     verbose=True
    # )

    best_rmse = float('inf')

    # Training loop
    print("\n" + "="*50)
    print("Starting Training with Cross-Modal Attention")
    print("="*50)

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(
            model=model,
            feature_extractors=feature_extractors,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=config.DEVICE
        )

        # Evaluate
        rmse, mae, r2 = evaluate(
            model=model,
            feature_extractors=feature_extractors,
            loader=val_loader,
            device=config.DEVICE
        )

        # Update scheduler
        # scheduler.step(rmse)

        # Print metrics
        print(f"Train Loss: {train_loss:.5f}")
        print(f"Val RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        # Save best model
        if rmse < best_rmse:
            best_rmse = rmse
            os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'config': {
                    'embed_dim': config.EMBED_DIM,
                    'num_heads': config.NUM_HEADS,
                    'dropout': config.DROPOUT
                }
            }, config.SAVE_PATH)
            print(f"✔ Saved best model (RMSE: {rmse:.4f})")

    print("\n" + "="*50)
    print(f"Training Complete! Best RMSE: {best_rmse:.4f}")
    print("="*50)

    return model


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    train_fusion_model()
