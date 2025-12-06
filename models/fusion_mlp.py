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




