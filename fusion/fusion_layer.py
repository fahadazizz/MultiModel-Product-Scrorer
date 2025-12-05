import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import numpy as np

import yaml
import os

class FusionLayer:
    def __init__(self, config_path='config/fusion_config.yaml'):
        # Load weights from config
        self.weights = self._load_config(config_path)

    def _load_config(self, config_path):
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    return config.get('fusion_weights', {
                        'sentiment': 0.5,
                        'image_confidence': 0.3,
                        'relevance': 0.2
                    })
            else:
                print(f"Warning: Config file {config_path} not found. Using default weights.")
                return {
                    'sentiment': 0.5,
                    'image_confidence': 0.3,
                    'relevance': 0.2
                }
        except Exception as e:
            print(f"Error loading config: {e}. Using default weights.")
            return {
                'sentiment': 0.5,
                'image_confidence': 0.3,
                'relevance': 0.2
            }
        
    def fuse(self, sentiment_scores, image_logits, image_embedding, text_embedding):
        """
        Fuse outputs to generate final score.
        """
        # 1. Sentiment Score (0.0 to 1.0)
        # Map: negative -> 0, neutral -> 0.5, positive -> 1
        sent_score = (
            sentiment_scores.get('negative', 0) * 0.0 + 
            sentiment_scores.get('neutral', 0) * 0.5 + 
            sentiment_scores.get('positive', 0) * 1.0
        )
        
        # 2. Image Confidence (Max probability)
        img_conf = torch.max(F.softmax(image_logits, dim=-1)).item()
        
        # 3. Relevance (Cosine Similarity normalized to 0-1)
        try:
            # Flatten and convert to numpy
            img_vec = image_embedding.detach().cpu().numpy().flatten() if isinstance(image_embedding, torch.Tensor) else image_embedding
            txt_vec = text_embedding.detach().cpu().numpy().flatten() if isinstance(text_embedding, torch.Tensor) else text_embedding
            
            # Cosine sim = 1 - distance. Normalize [-1, 1] -> [0, 1]
            relevance = (1 - cosine(img_vec, txt_vec) + 1) / 2
        except:
            relevance = 0.0
            
        # 4. Final Weighted Sum
        final_score = (
            self.weights['sentiment'] * sent_score +
            self.weights['image_confidence'] * img_conf +
            self.weights['relevance'] * relevance
        )
        
        return {
            'final_score': final_score,
            'sentiment_score': sent_score,
            'image_confidence_score': img_conf,
            'relevance_score': relevance
        }
