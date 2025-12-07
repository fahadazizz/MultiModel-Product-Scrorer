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
