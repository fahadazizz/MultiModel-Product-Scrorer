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
# FUSION MLP
# ==============================
# FusionMLP is now imported from models.fusion_mlp as MultimodalFusion
# We wrap it here if needed or just use it directly in the trainer.
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: Text embeddings (Batch, Seq_len_Q, Dim)
            key: Image embeddings (Batch, Seq_len_K, Dim)
            value: Image embeddings (Batch, Seq_len_K, Dim)
        Returns:
            attended_query: (Batch, Seq_len_Q, Dim)
        """
        # Multihead Attention
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # Residual + Norm
        output = self.norm(query + self.dropout(attn_output))
        return output

class MultimodalFusion(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.2):
        super().__init__()
        
        # Cross Attention: Text queries Image
        self.cross_attention = CrossModalAttention(embed_dim=input_dim, num_heads=8, dropout=dropout)
        
        # Fusion Network
        # Input to fusion is [Attended_Text_CLS, ViT_CLS] -> 768 + 768 = 1536
        fusion_input_dim = input_dim * 2 
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512 -> 256
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 64),         # 256 -> 64
            nn.ReLU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.fusion:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, vit_sequences, text_cls):
        """
        Args:
            vit_sequences: (Batch, Seq_len_Img, Dim) - e.g. (B, 197, 768)
            text_cls: (Batch, Dim) - Text CLS token - e.g. (B, 768)
        """
        device = text_cls.device
        
        # 1. Prepare Inputs
        # Text CLS needs to be sequence for attention: (B, 768) -> (B, 1, 768)
        if text_cls.dim() == 2:
            query = text_cls.unsqueeze(1)
        else:
            query = text_cls
            
        key = vit_sequences
        value = vit_sequences
        
        # 2. Cross-Modal Attention (Text queries Image)
        # Output: (B, 1, 768)
        attended_text = self.cross_attention(query, key, value)
        attended_text = attended_text.squeeze(1) # (B, 768)
        
        # 3. Get Image CLS token (assuming index 0 is CLS)
        # (B, 197, 768) -> (B, 768)
        image_cls = vit_sequences[:, 0, :]
        
        # 4. Concatenate
        fused_features = torch.cat([attended_text, image_cls], dim=1) # (B, 1536)
        
        # 5. MLP Score
        score = self.fusion(fused_features).squeeze(1) # (B)
        
        return score

# ==============================
# TRAINING FUNCTIONS
# ==============================
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        vit = batch["vit_features"].to(config.DEVICE)
        txt = batch["text_features"].to(config.DEVICE)
        target = batch["score"].to(config.DEVICE)

        opt.zero_grad()
        pred = model(vit, txt)
        loss = loss_fn(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

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
        train_loss = train_epoch(model, train_loader, opt, loss_fn)
        rmse, mae, r2 = evaluate(model, val_loader, loss_fn)
        scheduler.step(rmse)
        print(f"Train Loss: {train_loss:.5f}")
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



# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tqdm import tqdm
# from transformers import AutoTokenizer, RobertaModel, ViTImageProcessor, ViTModel
# import warnings
# warnings.filterwarnings('ignore')

# # Configuration
# class Config:
#     BATCH_SIZE = 64
#     LEARNING_RATE = 4e-4
#     NUM_EPOCHS = 20
#     HIDDEN_DIM = 256
#     DROPOUT_RATE = 0.3
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     MODEL_SAVE_PATH = 'models/fusionMLP_model.pth'
#     DATA_PATH = 'dataset/fusion_dataset.csv'
#     IMAGE_DIR = 'dataset/images/'

# config = Config()

# # 1. Feature Extractors (Pretrained Models)
# class FeatureExtractors:
#     def __init__(self):
#         print("Loading pretrained models...")
#         self.vit_model = ViTModel.from_pretrained('models/trained/finetuned_vit_fahad')
#         self.vit_feature_extractor = ViTImageProcessor.from_pretrained('models/trained/finetuned_vit_fahad',  use_safetensors=True)
        
#         self.roberta_model = RobertaModel.from_pretrained('models/trained/finetuned_roberta_fahad', use_safetensors=True)
#         self.roberta_tokenizer = AutoTokenizer.from_pretrained('models/trained/finetuned_roberta_fahad')
        
#         self.vit_model = self.vit_model.to(config.DEVICE)
#         self.roberta_model = self.roberta_model.to(config.DEVICE)
#         self.vit_model.eval()
#         self.roberta_model.eval()
        
#         print("Pretrained models loaded successfully!")
    
#     @torch.no_grad()
#     def get_vit_features(self, pixel_values):
#         outputs = self.vit_model(pixel_values)
#         return outputs.last_hidden_state[:, 0, :]  # CLS token

#     @torch.no_grad()
#     def get_roberta_features(self, input_ids, attention_mask):
#         outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
#         return outputs.last_hidden_state[:, 0, :]  # CLS token

# # 2. Pre-compute Embeddings Dataset
# class CachedEmbeddingDataset(Dataset):
#     def __init__(self, vit_embeddings, text_embeddings, scores):
#         self.vit_embeddings = vit_embeddings
#         self.text_embeddings = text_embeddings
#         self.scores = scores
    
#     def __len__(self):
#         return len(self.scores)
    
#     def __getitem__(self, idx):
#         return {
#             'vit_features': self.vit_embeddings[idx],
#             'text_features': self.text_embeddings[idx],
#             'score': self.scores[idx],
#             'original_score': 1 + self.scores[idx] * 9  # Convert back to 1-10 for metrics
#         }


# # 3. MultiModel Fusion
# class MultimodalFusion(nn.Module):
#     """FIXED architecture - No BatchNorm for single-sample inference"""
#     def __init__(self, input_dim=768, hidden_dim=256, dropout=0.2):
#         super().__init__()
#         self.fusion = nn.Sequential(
#             nn.Linear(input_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#             nn.Sigmoid()
#         )

#         # Initialize weights properly
#         for layer in self.fusion:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
#                 nn.init.zeros_(layer.bias)

#     def forward(self, vit_features, text_features):
#         fused = torch.cat([vit_features, text_features], dim=1)
#         return self.fusion(fused).squeeze(1)



# # 4. Pre-compute Embeddings Function
# def precompute_embeddings(dataset, feature_extractors, device):
#     """Pre-compute all embeddings to avoid repeated feature extraction"""
#     from torch.utils.data import DataLoader
    
#     # Create a temporary dataset without preprocessing for embedding extraction
#     class TempDataset(Dataset):
#         def __init__(self, df):
#             self.df = df
        
#         def __len__(self):
#             return len(self.df)
        
#         def __getitem__(self, idx):
#             return {
#                 'image_path': self.df.iloc[idx]['image_path'],
#                 'review': str(self.df.iloc[idx]['review']) if pd.notna(self.df.iloc[idx]['review']) else "No review",
#                 'score': float(self.df.iloc[idx]['score'])
#             }
    
#     temp_dataset = TempDataset(dataset)
#     temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
    
#     vit_embeddings = []
#     text_embeddings = []
#     scores = []
    
#     with torch.no_grad():
#         for batch in tqdm(temp_loader, desc="Pre-computing embeddings"):
#             # Process images
#             batch_images = []
#             for img_path in batch['image_path']:
#                 img_path = os.path.join(config.IMAGE_DIR, img_path)
#                 try:
#                     image = Image.open(img_path).convert('RGB')
#                     inputs = feature_extractors.vit_feature_extractor(
#                         images=image, return_tensors="pt"
#                     )
#                     batch_images.append(inputs['pixel_values'])
#                 except Exception as e:
#                     dummy = torch.zeros(1, 3, 224, 224)
#                     batch_images.append(dummy)
            
#             pixel_values = torch.cat(batch_images, dim=0).to(device)
#             vit_feats = feature_extractors.get_vit_features(pixel_values)
            
#             # Process texts
#             reviews = batch['review']
#             text_inputs = feature_extractors.roberta_tokenizer(
#                 reviews, padding=True, truncation=True, return_tensors="pt"
#             )
#             input_ids = text_inputs['input_ids'].to(device)
#             attention_mask = text_inputs['attention_mask'].to(device)
#             text_feats = feature_extractors.get_roberta_features(input_ids, attention_mask)
            
#             # Normalize scores to 0-1 range
#             batch_scores = torch.tensor([(s - 1) / 9.0 for s in batch['score']], dtype=torch.float32)
            
#             vit_embeddings.append(vit_feats.cpu())
#             text_embeddings.append(text_feats.cpu())
#             scores.append(batch_scores)
    
#     return (
#         torch.cat(vit_embeddings, dim=0),
#         torch.cat(text_embeddings, dim=0),
#         torch.cat(scores, dim=0)
#     )

# # 5. Training Functions
# def train_epoch(model, train_loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
    
#     for batch in tqdm(train_loader, desc='Training'):
#         vit_features = batch['vit_features'].to(device)
#         text_features = batch['text_features'].to(device)
#         targets = batch['score'].to(device)
        
#         optimizer.zero_grad()
#         outputs = model(vit_features, text_features)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     return total_loss / len(train_loader)

# def evaluate(model, val_loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for batch in val_loader:
#             vit_features = batch['vit_features'].to(device)
#             text_features = batch['text_features'].to(device)
#             targets = batch['score'].to(device)
#             original_targets = batch['original_score'].numpy()  # 1-10 scale
            
#             outputs = model(vit_features, text_features)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
            
#             # Convert predictions back to 1-10 scale for metrics
#             normalized_preds = outputs.cpu().numpy()
#             original_preds = 1 + normalized_preds * 9
            
#             all_preds.extend(original_preds)
#             all_targets.extend(original_targets)
    
#     avg_loss = total_loss / len(val_loader)
#     rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
#     mae = mean_absolute_error(all_targets, all_preds)
#     r2 = r2_score(all_targets, all_preds)  # CORRECT R² calculation
    
#     return avg_loss, rmse, mae, r2

# # 6. Main Training Function
# def train_multimodal_model():
#     print("Loading dataset...")
#     df = pd.read_csv(config.DATA_PATH)
    
#     # Split data
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
#     print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
#     # Initialize feature extractors
#     feature_extractors = FeatureExtractors()
    
#     # Pre-compute embeddings (one-time cost)
#     print("Pre-computing training embeddings...")
#     train_vit_embs, train_text_embs, train_scores = precompute_embeddings(train_df, feature_extractors, config.DEVICE)
    
#     print("Pre-computing validation embeddings...")
#     val_vit_embs, val_text_embs, val_scores = precompute_embeddings(val_df, feature_extractors, config.DEVICE)
    
#     # Create cached datasets
#     train_dataset = CachedEmbeddingDataset(train_vit_embs, train_text_embs, train_scores)
#     val_dataset = CachedEmbeddingDataset(val_vit_embs, val_text_embs, val_scores)
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
#     # Initialize model
#     model = MultimodalFusion(hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT_RATE).to(config.DEVICE)
    
#     # Loss and optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
#     best_val_rmse = float('inf')
    
#     # Training loop
#     print("\nStarting training...")
#     for epoch in range(config.NUM_EPOCHS):
#         print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
#         # Training
#         train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        
#         # Validation
#         val_loss, val_rmse, val_mae, val_r2 = evaluate(model, val_loader, criterion, config.DEVICE)
        
#         # Update scheduler
#         scheduler.step(val_loss)
        
#         # Save best model
#         if val_rmse < best_val_rmse:
#             best_val_rmse = val_rmse
#             os.makedirs('models', exist_ok=True)
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_rmse': val_rmse,
#                 'val_mae': val_mae,
#                 'val_r2': val_r2
#             }, config.MODEL_SAVE_PATH)
#             print(f"Saved best model with RMSE: {val_rmse:.4f}")
        
#         # Print metrics
#         print(f"Train Loss: {train_loss:.6f}")
#         print(f"Val Loss: {val_loss:.6f}")
#         print(f"Val RMSE: {val_rmse:.4f}")
#         print(f"Val MAE: {val_mae:.4f}")
#         print(f"Val R²: {val_r2:.4f}")
    
#     print(f"\nTraining completed! Best validation RMSE: {best_val_rmse:.4f}")
#     return model


# if __name__ == "__main__":
#     trained_model = train_multimodal_model()
#     print("\nTraining completed successfully!")

