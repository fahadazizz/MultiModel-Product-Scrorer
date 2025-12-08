import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from models.fusion_mlp import MultimodalFusion
import glob

from transformers import (
    AutoTokenizer,
    RobertaModel,
    ViTImageProcessor,
    ViTModel 
)

# =============================
# CONFIG
# =============================
class Config:
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 25
    DROPOUT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VIT_PATH = "models/trained/finetuned_vit_fahad"
    ROBERTA_PATH = "models/trained/finetuned_roberta_fahad"
    DATA_PATH = "dataset/products_augmented.csv"
    IMAGE_DIR = "dataset/" 
    SAVE_PATH = "models/trained/fusion_mlp_final.pth"

config = Config()

# ==============================
# FEATURE EXTRACTORS
# ==============================
class FeatureExtractors:
    def __init__(self):
        print("\nLoading pretrained models...")

        self.vit_model = ViTModel.from_pretrained(config.VIT_PATH).to(config.DEVICE).eval()
        self.vit_processor = ViTImageProcessor.from_pretrained(config.VIT_PATH, use_safetensors=True)

        self.roberta_model = RobertaModel.from_pretrained(config.ROBERTA_PATH, use_safetensors=True).to(config.DEVICE).eval()
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(config.ROBERTA_PATH)

        print("Models loaded successfully!")

    @torch.no_grad()
    def get_vit_embedding(self, pixel_values):
        pixel_values = pixel_values.to(config.DEVICE)
        outputs = self.vit_model(pixel_values, output_hidden_states=True)
        return outputs.last_hidden_state  # Return sequence: (B, 197, 768)

    @torch.no_grad()
    def get_text_embedding(self, input_ids, attention_mask):
        input_ids = input_ids.to(config.DEVICE)
        attention_mask = attention_mask.to(config.DEVICE)
        outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

# ==============================
# PRECOMPUTE EMBEDDINGS
# ==============================
def precompute_embeddings(df, fe):
    vit_embs, txt_embs, scores = [], [], []

    valid_indices = []
    
    loader = DataLoader(df.to_dict("records"), batch_size=16, shuffle=False)
    print("\nPrecomputing embeddings...")

    for i, batch in enumerate(tqdm(loader)):
        batch_images = []
        batch_valid_mask = []
        
        # Load and process images safely
        for img_path in batch["image_path"]:
            full_path = os.path.join(config.IMAGE_DIR, str(img_path).lstrip('/'))
            try:
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Image not found: {full_path}")
                    
                image = Image.open(full_path).convert("RGB")
                inputs = fe.vit_processor(images=image, return_tensors="pt")
                batch_images.append(inputs["pixel_values"])
                batch_valid_mask.append(True)
            except Exception as e:
                print(f"Skipping corrupt image {full_path}: {e}")
                batch_images.append(torch.zeros(1, 3, 224, 224)) # Placeholder to keep batch alignment for now
                batch_valid_mask.append(False)

        if not batch_images:
            continue
            
        pixel_values = torch.cat(batch_images, dim=0).to(config.DEVICE)
        
        # Only compute embeddings for valid images to save compute, but for simplicity here we compute all and mask later
        # Actually better to just compute all and filter before appending to lists
        with torch.no_grad():
            vit_feat = fe.get_vit_embedding(pixel_values).cpu()

        texts = [str(t) for t in batch["review_text"]]
        tokenized = fe.roberta_tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            txt_feat = fe.get_text_embedding(tokenized["input_ids"], tokenized["attention_mask"]).cpu()

        # Process scores and filter invalid
        for idx, valid in enumerate(batch_valid_mask):
            if not valid:
                continue
                
            s = batch["score"][idx]
            if pd.isna(s):
                continue
                
            norm_score = (float(s) - 1) / 9.0
            
            vit_embs.append(vit_feat[idx].unsqueeze(0))
            txt_embs.append(txt_feat[idx].unsqueeze(0))
            scores.append(torch.tensor([norm_score]))

    if not vit_embs:
        raise ValueError("No valid data found after precomputing embeddings!")

    return torch.cat(vit_embs, dim=0), torch.cat(txt_embs, dim=0), torch.cat(scores, dim=0)

def precompute_external_embeddings(fe):
    """
    Load images from dataset/images/ for external negative sampling.
    """
    print("\nPrecomputing external image embeddings (Distractors)...")
    image_paths = glob.glob("dataset/images/**/*.jpg", recursive=True) + glob.glob("dataset/images/**/*.png", recursive=True)
    
    if not image_paths:
        print("Warning: No external images found in dataset/images/")
        return None

    vit_embs = []
    # Process in batches to save memory during inference
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                inputs = fe.vit_processor(images=img, return_tensors="pt")
                batch_images.append(inputs["pixel_values"])
            except Exception as e:
                print(f"Error loading {p}: {e}")
                continue
        
        if batch_images:
            pixel_values = torch.cat(batch_images, dim=0).to(config.DEVICE)
            with torch.no_grad():
                vit_feat = fe.get_vit_embedding(pixel_values).cpu()
            vit_embs.append(vit_feat)
    
    if vit_embs:
        return torch.cat(vit_embs, dim=0)
    return None

# ==============================
# DATASET
# ==============================
class FusionDataset(Dataset):
    def __init__(self, vit_embs, txt_embs, scores):
        self.vit = vit_embs
        self.txt = txt_embs
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, i):
        return {
            "vit_features": self.vit[i],
            "text_features": self.txt[i],
            "score": self.scores[i],
            "original_score": 1 + self.scores[i] * 9
        }


# ==============================
# TRAINING FUNCTIONS
# ==============================
def train_epoch(model, loader, opt, rec_loss_fn, rel_loss_fn, external_embs=None):
    model.train()
    total_loss = 0
    
    # Pre-move external embeddings to device if possible or sample batches
    if external_embs is not None:
        num_ext = external_embs.size(0)
    
    for batch in tqdm(loader, desc="Training"):
        vit = batch["vit_features"].to(config.DEVICE)
        txt = batch["text_features"].to(config.DEVICE)
        target_score = batch["score"].to(config.DEVICE)
        
        batch_size = vit.size(0)

        opt.zero_grad()
        
        # ==========================
        # 1. Positive Pair (Real)
        # ==========================
        pred_score, pred_rel = model(vit, txt)
        
        # Loss: Score matches target, Relevance = 1
        loss_rec = rec_loss_fn(pred_score, target_score)
        loss_rel = rel_loss_fn(pred_rel, torch.ones_like(pred_rel))
        
        loss_pos = loss_rec + loss_rel
        
        # ==========================
        # 2. Negative Pair (Internal - Mismatched Text)
        # ==========================
        txt_neg = torch.roll(txt, shifts=1, dims=0)
        pred_score_neg, pred_rel_neg = model(vit, txt_neg)
        
        # Loss: Score = 0 (Bad match), Relevance = 0
        loss_rec_neg = rec_loss_fn(pred_score_neg, torch.zeros_like(pred_score_neg))
        loss_rel_neg = rel_loss_fn(pred_rel_neg, torch.zeros_like(pred_rel_neg))
        
        loss_neg_int = loss_rec_neg + loss_rel_neg
        
        # ==========================
        # 3. Negative Pair (External - Random Image)
        # ==========================
        loss_neg_ext = 0
        if external_embs is not None:
            # Sample random external images
            indices = torch.randint(0, num_ext, (batch_size,))
            full_b_indices = indices # In case batch_size > num_ext (rare if lots of images)
            
            ext_vit = external_embs[indices].to(config.DEVICE)
            
            # Use current text (it doesn't match the external image)
            pred_score_ext, pred_rel_ext = model(ext_vit, txt)
            
            loss_rec_ext = rec_loss_fn(pred_score_ext, torch.zeros_like(pred_score_ext))
            loss_rel_ext = rel_loss_fn(pred_rel_ext, torch.zeros_like(pred_rel_ext))
            
            loss_neg_ext = loss_rec_ext + loss_rel_ext

        # ==========================
        # Total Loss
        # ==========================
        # Balanced: 50% Positive, 25% Neg Int, 25% Neg Ext
        loss = loss_pos + 0.5 * loss_neg_int + 0.5 * loss_neg_ext
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds_score, trues_score = [], []
    preds_rel, trues_rel = [], []
    
    with torch.no_grad():
        for batch in loader:
            vit = batch["vit_features"].to(config.DEVICE)
            txt = batch["text_features"].to(config.DEVICE)
            score = batch["score"].to(config.DEVICE)
            
            # Forward pass
            # Note: For evaluation on the positive dataset, we expect Relevance = 1.0
            pred_score_raw, pred_rel_raw = model(vit, txt)
            
            # Score Metrics
            preds_score.extend((1 + pred_score_raw.cpu().numpy() * 9))
            trues_score.extend(batch["original_score"].numpy())
            
            # Relevance Metrics (Threshold 0.5)
            # Since the validation set currently only contains Positive pairs (from split),
            # Trues for relevance are all 1s. 
            # To get a valid relevance metric, we should ideally inject negatives.
            # However, for simply checking if the model 'accepts' valid pairs:
            preds_rel.extend((pred_rel_raw.cpu().numpy() > 0.5).astype(int))
            trues_rel.extend(np.ones(len(pred_rel_raw))) # We assume validation set is all positive pairs

    # Score Metrics
    rmse = np.sqrt(mean_squared_error(trues_score, preds_score))
    mae = mean_absolute_error(trues_score, preds_score)
    r2 = r2_score(trues_score, preds_score)
    
    # Relevance Metrics (Precision/Recall might be boring if all trues are 1, but Accuracy is useful)
    accuracy = accuracy_score(trues_rel, preds_rel)
    
    return rmse, mae, r2, accuracy

# ==============================
# TRAINER
# ==============================
def train_fusion_model():
    print("Loading dataset...")
    df = pd.read_csv(config.DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    fe = FeatureExtractors()
    train_vit, train_txt, train_scores = precompute_embeddings(train_df, fe)
    val_vit, val_txt, val_scores = precompute_embeddings(val_df, fe)
    
    # Precompute External Embeddings
    external_embs = precompute_external_embeddings(fe)

    train_ds = FusionDataset(train_vit, train_txt, train_scores)
    val_ds = FusionDataset(val_vit, val_txt, val_scores)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = MultimodalFusion(dropout=config.DROPOUT).to(config.DEVICE)
    rec_loss_fn = nn.MSELoss()
    rel_loss_fn = nn.BCELoss()
    
    opt = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_rmse = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, opt, rec_loss_fn, rel_loss_fn, external_embs)
        rmse, mae, r2, rel_acc = evaluate(model, val_loader)
        scheduler.step(rmse)
        print(f"Train Loss: {train_loss:.5f}")
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Rel Acc (Pos): {rel_acc:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"✔ Saved best model (RMSE: {rmse:.4f})")

    print("\nTraining Complete!")
    return model

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")
    train_fusion_model()


