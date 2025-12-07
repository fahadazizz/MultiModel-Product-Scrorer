import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from models.fusion_mlp import MultimodalFusion

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
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 25
    DROPOUT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VIT_PATH = "models/trained/finetuned_vit_fahad"
    ROBERTA_PATH = "models/trained/finetuned_roberta_fahad"
    DATA_PATH = "dataset/products_processed.csv"
    IMAGE_DIR = "dataset/multimodel_images/" 
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

    loader = DataLoader(df.to_dict("records"), batch_size=16, shuffle=False)
    print("\nPrecomputing embeddings...")

    for batch in tqdm(loader):
        batch_images = []
        for img_path in batch["image_path"]:
            # Image paths in csv might start with /, so we strip it.
            full_path = os.path.join(config.IMAGE_DIR, str(img_path).lstrip('/'))
            image = Image.open(full_path).convert("RGB")
            inputs = fe.vit_processor(images=image, return_tensors="pt")
            batch_images.append(inputs["pixel_values"])

        pixel_values = torch.cat(batch_images, dim=0).to(config.DEVICE)
        vit_feat = fe.get_vit_embedding(pixel_values).cpu()

        texts = [str(t) for t in batch["review_text"]]
        tokenized = fe.roberta_tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        txt_feat = fe.get_text_embedding(tokenized["input_ids"], tokenized["attention_mask"]).cpu()

        norm_scores = torch.tensor([(float(s) - 1) / 9.0 for s in batch["score"]])

        vit_embs.append(vit_feat)
        txt_embs.append(txt_feat)
        scores.append(norm_scores)

    return torch.cat(vit_embs, dim=0), torch.cat(txt_embs, dim=0), torch.cat(scores, dim=0)

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
def info_nce_loss(features_a, features_b, temperature=0.07):
    """
    Computes InfoNCE loss.
    Args:
        features_a: (B, Dim) - e.g. Image embeddings
        features_b: (B, Dim) - e.g. Text embeddings
    Returns:
        loss: Scalar tensor
    """
    # Normalize features if not already done (though model does it)
    features_a = F.normalize(features_a, p=2, dim=1)
    features_b = F.normalize(features_b, p=2, dim=1)
    
    # Similarity matrix: (B, B)
    logits = torch.matmul(features_a, features_b.T) / temperature
    
    # Labels: diagonal elements are positives
    batch_size = features_a.shape[0]
    labels = torch.arange(batch_size).to(features_a.device)
    
    # Cross Entropy Loss
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0
    mse_loss = 0
    con_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        vit = batch["vit_features"].to(config.DEVICE)
        txt = batch["text_features"].to(config.DEVICE)
        target = batch["score"].to(config.DEVICE)

        opt.zero_grad()
        
        # Used forward_train to get projections
        pred, img_proj, txt_proj = model.forward_train(vit, txt)
        
        # 1. MSE Loss (Score)
        loss_mse = loss_fn(pred, target)
        
        # Check for NaN in MSE
        if torch.isnan(loss_mse):
            print(f"❌ NaN MSE loss detected! Skipping batch.")
            continue

        # 2. Contrastive Loss (Relevance)
        loss_con = info_nce_loss(img_proj, txt_proj)
        
        # Combine: Score is main goal, Relevance is auxiliary.
        # Alpha: 0.5 seems reasonable to balance learning representation vs scoring.
        total_batch_loss = loss_mse + 0.5 * loss_con
        
        if torch.isnan(total_batch_loss):
            print(f"❌ NaN Total Ioss detected! Skipping batch.")
            continue
            
        total_batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        total_loss += total_batch_loss.item()
        mse_loss += loss_mse.item()
        con_loss += loss_con.item()
        
    avg_loss = total_loss / len(loader)
    avg_mse = mse_loss / len(loader)
    avg_con = con_loss / len(loader)
    
    return avg_loss, avg_mse, avg_con

def evaluate(model, loader, loss_fn):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            vit = batch["vit_features"].to(config.DEVICE)
            txt = batch["text_features"].to(config.DEVICE)
            score = batch["score"].to(config.DEVICE)
            pred = model(vit, txt)
            preds.extend((1 + pred.cpu().numpy() * 9))
            trues.extend(batch["original_score"].numpy())
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return rmse, mae, r2

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

    train_ds = FusionDataset(train_vit, train_txt, train_scores)
    val_ds = FusionDataset(val_vit, val_txt, val_scores)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = MultimodalFusion(dropout=config.DROPOUT).to(config.DEVICE)
    loss_fn = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_rmse = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss, train_mse, train_con = train_epoch(model, train_loader, opt, loss_fn)
        rmse, mae, r2 = evaluate(model, val_loader, loss_fn)
        scheduler.step(rmse)
        print(f"Train Loss: {train_loss:.5f} (MSE: {train_mse:.5f}, Con: {train_con:.5f})")
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

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


