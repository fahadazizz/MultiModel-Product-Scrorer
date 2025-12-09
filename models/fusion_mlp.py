import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


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
    Multimodal Fusion with Cross-Modal Attention and Relevance Scoring.
    
    Pipeline:
    1. Cross-Modal Attention: Text attends to Image
    2. Compute Relevance Score:
       - Alignment Score: cosine(pooled_attended_text, pooled_image)
       - Shift Score: 1 - cosine(original_text, attended_text) → how much attention changed text
       - Combined: w1 * alignment + w2 * shift
    3. Pooling: Mean pool attended text + CLS from image
    4. Concatenation: [pooled_text, pooled_image] -> (B, 2*D)
    5. MLP Score Head: Produces recommendation score
    """
    
    # Relevance thresholds and weights
    RELEVANCE_THRESHOLD = 0.4
    ALIGNMENT_WEIGHT = 0.75
    SHIFT_WEIGHT = 0.25      
    
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
    
    def _compute_relevance(
        self,
        original_text_pooled: torch.Tensor,
        attended_text_pooled: torch.Tensor,
        image_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute relevance score using alignment and shift metrics.
        
        Args:
            original_text_pooled: Original text embedding before attention (B, D)
            attended_text_pooled: Text embedding after cross-attention (B, D)
            image_pooled: Image CLS embedding (B, D)
            
        Returns:
            relevance_score: Combined relevance score (B,) in range [0, 1]
            details: Dictionary with alignment_score, shift_score
        """
        
        alignment_score = F.cosine_similarity(attended_text_pooled, image_pooled, dim=1)
        alignment_score = (alignment_score + 1) / 2
        
        shift_similarity = F.cosine_similarity(original_text_pooled, attended_text_pooled, dim=1)
        shift_score = 1 - shift_similarity
        shift_score = shift_score.clamp(0, 1)
        
        relevance_score = (
            self.ALIGNMENT_WEIGHT * alignment_score + 
            self.SHIFT_WEIGHT * shift_score
        )
        
        details = {
            'alignment_score': alignment_score,
            'shift_score': shift_score
        }
        
        return relevance_score, details

    def forward(
        self,
        image_seq: torch.Tensor,
        text_seq: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        return_relevance: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with optional relevance scoring.
        
        Args:
            image_seq: Image sequence embeddings (B, N_patches+1, D)
            text_seq: Text sequence embeddings (B, seq_len, D)
            text_attention_mask: Mask for text tokens (B, seq_len)
            return_relevance: If True, returns (score, relevance_dict)
            
        Returns:
            score: Recommendation score (B,) in range [0, 1]
            OR (score, relevance_dict) if return_relevance=True
        """
        device = next(self.parameters()).device
        image_seq = image_seq.to(device)
        text_seq = text_seq.to(device)
        
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(device)
            mask_expanded = text_attention_mask.unsqueeze(-1).float()
            original_text_pooled = (text_seq * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            original_text_pooled = text_seq.mean(dim=1)
        
        # Cross-Modal Attention
        attended_text, attn_weights = self.cross_attention(
            text_emb=text_seq,
            image_emb=image_seq
        )
        
        # Pool attended text
        if text_attention_mask is not None:
            pooled_text = (attended_text * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled_text = attended_text.mean(dim=1)
        
        # Pool image (CLS token)
        pooled_image = image_seq[:, 0, :]
        
        # Compute Relevance Score
        relevance_score, relevance_details = self._compute_relevance(
            original_text_pooled=original_text_pooled,
            attended_text_pooled=pooled_text,
            image_pooled=pooled_image
        )
        
        # Concatenate and compute final score
        fused = torch.cat([pooled_text, pooled_image], dim=1)
        score = self.score_head(fused).squeeze(-1)
        
        if return_relevance:
            return score, {
                'relevance_score': relevance_score,
                'alignment_score': relevance_details['alignment_score'],
                'shift_score': relevance_details['shift_score'],
                'is_relevant': relevance_score >= self.RELEVANCE_THRESHOLD,
                'attention_weights': attn_weights
            }
        
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
