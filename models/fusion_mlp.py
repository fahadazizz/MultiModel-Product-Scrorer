import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    """FIXED architecture - No BatchNorm for single-sample inference"""
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights properly
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, vit_features, text_features):
        fused = torch.cat([vit_features, text_features], dim=1)
        return self.fusion(fused).squeeze(1)

# import torch
# import torch.nn as nn

# class FusionMLP(nn.Module):
#     def __init__(self, input_dim=768, hidden_dim=512, dropout=0.3):
#         super().__init__()
#         self.vit_proj = nn.Linear(input_dim, hidden_dim)
#         self.text_proj = nn.Linear(input_dim, hidden_dim)
        
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),  
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1) 
#         )
        
#         # For regression to 1-10 range
#         self.output_activation = nn.Sigmoid() 
    
#     def forward(self, vit_features, text_features):
#         vit_proj = self.vit_proj(vit_features) 
#         text_proj = self.text_proj(text_features)  
        
#         # Concatenate modalities
#         fused = torch.cat([vit_proj, text_proj], dim=1) 
        
#         raw_score = self.fusion(fused) 
        
#         normalized_score = self.output_activation(raw_score) 
#         final_score = 1 + 9 * normalized_score  
        
#         return final_score



# # class AttentionFusion(nn.Module):
# #     def __init__(self, input_dim=768, hidden_dim=256):
# #         super().__init__()
# #         self.vit_proj = nn.Linear(input_dim, hidden_dim)
# #         self.text_proj = nn.Linear(input_dim, hidden_dim)
        
# #         # Attention mechanism to weight modalities
# #         self.attention = nn.Sequential(
# #             nn.Linear(hidden_dim * 2, hidden_dim),
# #             nn.Tanh(),
# #             nn.Linear(hidden_dim, 2),  # Weights for [vit, text]
# #             nn.Softmax(dim=1)  # Normalize to [0,1] weights
# #         )
        
# #         # Regression head
# #         self.regressor = nn.Sequential(
# #             nn.Linear(hidden_dim * 2, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, 1)
# #         )
        
# #         self.output_activation = nn.Sigmoid()
    
# #     def forward(self, vit_features, text_features):
# #         vit_proj = self.vit_proj(vit_features)  # [batch, hidden_dim]
# #         text_proj = self.text_proj(text_features)  # [batch, hidden_dim]
        
# #         # Calculate attention weights
# #         combined = torch.cat([vit_proj, text_proj], dim=1)
# #         weights = self.attention(combined)  # [batch, 2] - weights for each modality
        
# #         # Apply weights
# #         weighted_vit = weights[:, 0].unsqueeze(1) * vit_proj
# #         weighted_text = weights[:, 1].unsqueeze(1) * text_proj
        
# #         # Fuse weighted features
# #         fused = torch.cat([weighted_vit, weighted_text], dim=1)
# #         raw_score = self.regressor(fused)
        
# #         # Scale to 1-10
# #         return 1 + 9 * self.output_activation(raw_score)