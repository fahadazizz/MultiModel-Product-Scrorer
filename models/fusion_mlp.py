import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Layer.
    Q = Text embeddings, K = V = Image embeddings
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        image_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        attended_output, attention_weights = self.multihead_attn(
            query=text_emb,
            key=image_emb,
            value=image_emb,
            key_padding_mask=image_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        attended_features = self.layer_norm(text_emb + self.dropout(attended_output))
        
        return attended_features, attention_weights


class MultimodalFusionWithAttention(nn.Module):
    """
    Multimodal Fusion with Cross-Modal Attention.
    
    Pipeline:
    1. Cross-Modal Attention: Text attends to Image
    2. Pooling: Mean pool attended text + CLS from image
    3. Concatenation: [pooled_text, pooled_image] -> (B, 2*D)
    4. MLP Score Head: Produces recommendation score
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_hidden_dims: Tuple[int, ...] = (512, 128, 32),
        dropout: float = 0.2,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.cross_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        
        layers = []
        input_dim = embed_dim * 2
        
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.score_head = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.score_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        image_seq: torch.Tensor,
        text_seq: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        device = next(self.parameters()).device
        image_seq = image_seq.to(device)
        text_seq = text_seq.to(device)
        
        attended_text, attn_weights = self.cross_attention(
            text_emb=text_seq,
            image_emb=image_seq
        )
        
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(device)
            mask_expanded = text_attention_mask.unsqueeze(-1).float()
            pooled_text = (attended_text * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled_text = attended_text.mean(dim=1)
        
        pooled_image = image_seq[:, 0, :]
        
        fused = torch.cat([pooled_text, pooled_image], dim=1)
        
        score = self.score_head(fused).squeeze(-1)
        
        return score
    
    def get_attention_weights(
        self,
        image_seq: torch.Tensor,
        text_seq: torch.Tensor
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        _, attn_weights = self.cross_attention(
            text_emb=text_seq.to(device),
            image_emb=image_seq.to(device)
        )
        return attn_weights


# Legacy class for backward compatibility
class MultimodalFusion(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        input_dim = 1536
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, vit, txt):
        vit = vit.to(next(self.parameters()).device)
        txt = txt.to(next(self.parameters()).device)
        vit = F.layer_norm(vit, vit.shape[1:])
        txt = F.layer_norm(txt, txt.shape[1:])
        fused = torch.cat([vit, txt], dim=1)
        output = self.fusion(fused).squeeze(1)
        return output
