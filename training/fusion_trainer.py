import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
# import matplotlib.pyplot as plt
from transformers import AutoTokenizer, RobertaModel, ViTImageProcessor, ViTModel
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 20
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.2
    IMAGE_SIZE = 224
    MAX_TEXT_LENGTH = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_SAVE_PATH = 'models/fusionMLP_model.pth'
    DATA_PATH = 'dataset/temp_fusion_dataset.csv'
    IMAGE_DIR = 'dataset/images/'

config = Config()

# 1. Feature Extractors (Pretrained Models)
class FeatureExtractors:
    def __init__(self):
        print("Loading pretrained models...")
        self.vit_model = ViTModel.from_pretrained('models/finetuned_vit_fahad')
        self.vit_feature_extractor = ViTImageProcessor.from_pretrained('models/finetuned_vit_fahad',  use_safetensors=True)
        
        self.roberta_model = RobertaModel.from_pretrained('models/finetuned_roberta_fahad', use_safetensors=True)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('models/finetuned_roberta_fahad')
        
        self.vit_model = self.vit_model.to(config.DEVICE)
        self.roberta_model = self.roberta_model.to(config.DEVICE)
        self.vit_model.eval()
        self.roberta_model.eval()
        
        print("Pretrained models loaded successfully!")
    
    @torch.no_grad()
    def get_vit_features(self, pixel_values):
        outputs = self.vit_model(pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    @torch.no_grad()
    def get_roberta_features(self, input_ids, attention_mask):
        outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

# 2. Pre-compute Embeddings Dataset
class CachedEmbeddingDataset(Dataset):
    def __init__(self, vit_embeddings, text_embeddings, scores):
        self.vit_embeddings = vit_embeddings
        self.text_embeddings = text_embeddings
        self.scores = scores
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        return {
            'vit_features': self.vit_embeddings[idx],
            'text_features': self.text_embeddings[idx],
            'score': self.scores[idx],
            'original_score': 1 + self.scores[idx] * 9  # Convert back to 1-10 for metrics
        }

# 3. Fusion Model Architecture
class MultimodalFusion(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.2):
        super().__init__()
        # Fusion network (simplified for speed)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, vit_features, text_features):
        fused = torch.cat([vit_features, text_features], dim=1)
        return self.fusion(fused).squeeze(1)

# 4. Pre-compute Embeddings Function
def precompute_embeddings(dataset, feature_extractors, device):
    """Pre-compute all embeddings to avoid repeated feature extraction"""
    from torch.utils.data import DataLoader
    
    # Create a temporary dataset without preprocessing for embedding extraction
    class TempDataset(Dataset):
        def __init__(self, df):
            self.df = df
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            return {
                'image_path': self.df.iloc[idx]['image_path'],
                'review': str(self.df.iloc[idx]['review']) if pd.notna(self.df.iloc[idx]['review']) else "No review",
                'score': float(self.df.iloc[idx]['score'])
            }
    
    temp_dataset = TempDataset(dataset)
    temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
    
    vit_embeddings = []
    text_embeddings = []
    scores = []
    
    with torch.no_grad():
        for batch in tqdm(temp_loader, desc="Pre-computing embeddings"):
            # Process images
            batch_images = []
            for img_path in batch['image_path']:
                img_path = os.path.join(config.IMAGE_DIR, img_path)
                try:
                    image = Image.open(img_path).convert('RGB')
                    inputs = feature_extractors.vit_feature_extractor(
                        images=image, return_tensors="pt", size={"height": config.IMAGE_SIZE, "width": config.IMAGE_SIZE}
                    )
                    batch_images.append(inputs['pixel_values'])
                except:
                    # Fallback: zero tensor
                    batch_images.append(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
            
            pixel_values = torch.cat(batch_images, dim=0).to(device)
            vit_feats = feature_extractors.get_vit_features(pixel_values)
            
            # Process texts
            reviews = batch['review']
            text_inputs = feature_extractors.roberta_tokenizer(
                reviews, padding=True, truncation=True, max_length=config.MAX_TEXT_LENGTH, return_tensors="pt"
            )
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            text_feats = feature_extractors.get_roberta_features(input_ids, attention_mask)
            
            # Normalize scores to 0-1 range
            batch_scores = torch.tensor([(s - 1) / 9.0 for s in batch['score']], dtype=torch.float32)
            
            vit_embeddings.append(vit_feats.cpu())
            text_embeddings.append(text_feats.cpu())
            scores.append(batch_scores)
    
    return (
        torch.cat(vit_embeddings, dim=0),
        torch.cat(text_embeddings, dim=0),
        torch.cat(scores, dim=0)
    )

# 5. Training Functions
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        vit_features = batch['vit_features'].to(device)
        text_features = batch['text_features'].to(device)
        targets = batch['score'].to(device)
        
        optimizer.zero_grad()
        outputs = model(vit_features, text_features)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            vit_features = batch['vit_features'].to(device)
            text_features = batch['text_features'].to(device)
            targets = batch['score'].to(device)
            original_targets = batch['original_score'].numpy()  # 1-10 scale
            
            outputs = model(vit_features, text_features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Convert predictions back to 1-10 scale for metrics
            normalized_preds = outputs.cpu().numpy()
            original_preds = 1 + normalized_preds * 9
            
            all_preds.extend(original_preds)
            all_targets.extend(original_targets)
    
    avg_loss = total_loss / len(val_loader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)  # CORRECT R² calculation
    
    return avg_loss, rmse, mae, r2

# 6. Main Training Function
def train_multimodal_model():
    print("Loading dataset...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Initialize feature extractors
    feature_extractors = FeatureExtractors()
    
    # Pre-compute embeddings (one-time cost)
    print("Pre-computing training embeddings...")
    train_vit_embs, train_text_embs, train_scores = precompute_embeddings(train_df, feature_extractors, config.DEVICE)
    
    print("Pre-computing validation embeddings...")
    val_vit_embs, val_text_embs, val_scores = precompute_embeddings(val_df, feature_extractors, config.DEVICE)
    
    # Create cached datasets
    train_dataset = CachedEmbeddingDataset(train_vit_embs, train_text_embs, train_scores)
    val_dataset = CachedEmbeddingDataset(val_vit_embs, val_text_embs, val_scores)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MultimodalFusion(hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT_RATE).to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_rmse = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        
        # Validation
        val_loss, val_rmse, val_mae, val_r2 = evaluate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2
            }, config.MODEL_SAVE_PATH)
            print(f"Saved best model with RMSE: {val_rmse:.4f}")
        
        # Print metrics
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val RMSE: {val_rmse:.4f}")
        print(f"Val MAE: {val_mae:.4f}")
        print(f"Val R²: {val_r2:.4f}")
    
    print(f"\nTraining completed! Best validation RMSE: {best_val_rmse:.4f}")
    return model

# 7. Inference Function
def predict_score(model, feature_extractors, image_path, review_text, device=config.DEVICE):
    """Predict recommendation score for a single product"""
    model.eval()
    
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    image_inputs = feature_extractors.vit_feature_extractor(
        images=image, return_tensors="pt", size={"height": config.IMAGE_SIZE, "width": config.IMAGE_SIZE}
    )
    pixel_values = image_inputs['pixel_values'].to(device)
    
    # Preprocess text
    text_inputs = feature_extractors.roberta_tokenizer(
        review_text, padding='max_length', truncation=True, max_length=config.MAX_TEXT_LENGTH, return_tensors="pt"
    )
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)
    
    # Extract features and predict
    with torch.no_grad():
        vit_features = feature_extractors.get_vit_features(pixel_values)
        text_features = feature_extractors.get_roberta_features(input_ids, attention_mask)
        normalized_pred = model(vit_features, text_features).cpu().item()
        final_score = 1 + normalized_pred * 9  # Convert back to 1-10 scale
    
    return final_score

if __name__ == "__main__":
    trained_model = train_multimodal_model()
    print("\nTraining completed successfully!")