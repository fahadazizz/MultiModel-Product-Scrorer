import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine

class FusionLayer:
    def __init__(self, weights={'sentiment': 0.4, 'image_confidence': 0.3, 'relevance': 0.3}):
        """
        Initialize the Fusion Layer.
        
        Args:
            weights: Dictionary of weights for each component. Must sum to 1.0.
        """
        self.weights = weights
        # Normalize weights just in case
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
        
    def fuse(self, sentiment_scores, image_logits, image_embedding, text_embedding):
        """
        Fuse the outputs from different models to generate a final recommendation score.
        
        Args:
            sentiment_scores: Dictionary of sentiment scores (negative, neutral, positive)
            image_logits: Logits from the image classifier
            image_embedding: Embedding vector from the image classifier
            text_embedding: Embedding vector from the text embedder
            
        Returns:
            Dictionary containing final score and intermediate scores.
        """
        # 1. Process Sentiment Score
        # We want a score from 0 (negative) to 1 (positive).
        # Assuming sentiment_scores has 'negative', 'neutral', 'positive'
        # We can use a weighted sum: 0*neg + 0.5*neu + 1*pos
        sentiment_score = self.compute_sentiment_score(sentiment_scores)
        
        # 2. Process Image Confidence Score
        # High confidence in classification (regardless of class) might indicate a clear product image.
        # However, for recommendation, we might want to know if the image is "good" or matches the text.
        # For now, let's use the max probability as a confidence metric.
        image_confidence_score = self.compute_image_confidence_score(image_logits)
        
        # 3. Process Relevance Score (Image-Text Alignment)
        # Cosine similarity between image and text embeddings.
        relevance_score = self.compute_relevance_score(image_embedding, text_embedding)
        
        # 4. Weighted Fusion
        final_score = (
            self.weights['sentiment'] * sentiment_score +
            self.weights['image_confidence'] * image_confidence_score +
            self.weights['relevance'] * relevance_score
        )
        
        return {
            'final_score': final_score,
            'sentiment_score': sentiment_score,
            'image_confidence_score': image_confidence_score,
            'relevance_score': relevance_score
        }
    
    def compute_sentiment_score(self, scores):
        """
        Convert sentiment distribution to a single 0-1 score.
        """
        # Map: negative -> 0.0, neutral -> 0.5, positive -> 1.0
        score = (scores.get('negative', 0) * 0.0 + 
                 scores.get('neutral', 0) * 0.5 + 
                 scores.get('positive', 0) * 1.0)
        return score

    def compute_image_confidence_score(self, logits):
        """
        Compute confidence score from image logits.
        """
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()
        return max_prob

    def compute_relevance_score(self, image_emb, text_emb):
        """
        Compute cosine similarity between image and text embeddings.
        """
        # Ensure they are 1D arrays
        if isinstance(image_emb, torch.Tensor):
            image_emb = image_emb.detach().cpu().numpy().flatten()
        if isinstance(text_emb, torch.Tensor):
            text_emb = text_emb.detach().cpu().numpy().flatten()
            
        # Cosine similarity is 1 - cosine distance
        # Range of cosine similarity is -1 to 1. We want 0 to 1.
        # So we can normalize: (sim + 1) / 2
        
        try:
            sim = 1 - cosine(image_emb, text_emb)
            norm_sim = (sim + 1) / 2
            return norm_sim
        except ValueError:
            # Handle zero vectors
            return 0.0
