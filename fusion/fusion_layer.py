import torch
import torch.nn.functional as F

class FusionLayer:
    def __init__(self, 
                 sentiment_weight=0.4, 
                 image_confidence_weight=0.3, 
                 relevance_weight=0.3):
        """
        Initialize fusion layer with weights for different components.
        
        Args:
            sentiment_weight: Weight for sentiment score
            image_confidence_weight: Weight for image classification confidence
            relevance_weight: Weight for image-text relevance score
        """
        self.sentiment_weight = sentiment_weight
        self.image_confidence_weight = image_confidence_weight
        self.relevance_weight = relevance_weight
        
        # Normalize weights
        total = sentiment_weight + image_confidence_weight + relevance_weight
        self.sentiment_weight /= total
        self.image_confidence_weight /= total
        self.relevance_weight /= total

    def compute_sentiment_score(self, sentiment_scores):
        """
        Convert sentiment scores to a normalized score.
        Args:
            sentiment_scores: Dict with keys 'negative', 'neutral', 'positive'
        Returns:
            Normalized score between 0 and 1
        """
        # Weighted scoring: negative=-1, neutral=0, positive=1, then normalize to 0-1
        score = (sentiment_scores['negative'] * -1.0 + 
                 sentiment_scores['neutral'] * 0.0 + 
                 sentiment_scores['positive'] * 1.0)
        # score is now between -1 and 1, normalize to 0-1
        normalized_score = (score + 1.0) / 2.0
        return normalized_score

    def compute_image_confidence_score(self, image_logits):
        """
        Compute confidence score from image classification logits.
        Args:
            image_logits: Tensor of logits from image classifier
        Returns:
            Confidence score (max probability)
        """
        probs = F.softmax(image_logits, dim=-1)
        max_prob = torch.max(probs).item()
        return max_prob

    def compute_relevance_score(self, image_embedding, text_embedding):
        """
        Compute relevance between image and text using cosine similarity.
        Args:
            image_embedding: Tensor of image embedding
            text_embedding: Tensor of text embedding
        Returns:
            Similarity score between 0 and 1
        """
        similarity = F.cosine_similarity(image_embedding, text_embedding)
        # Cosine similarity is between -1 and 1, normalize to 0-1
        normalized_similarity = (similarity + 1.0) / 2.0
        return normalized_similarity.item()

    def fuse(self, sentiment_scores, image_logits, image_embedding, text_embedding):
        """
        Fuse all scores to produce final recommendation score.
        
        Args:
            sentiment_scores: Dict with sentiment scores
            image_logits: Tensor from image classifier
            image_embedding: Tensor of image embedding
            text_embedding: Tensor of text embedding
        
        Returns:
            Final recommendation score between 0 and 1
        """
        sentiment_score = self.compute_sentiment_score(sentiment_scores)
        image_conf_score = self.compute_image_confidence_score(image_logits)
        relevance_score = self.compute_relevance_score(image_embedding, text_embedding)
        
        final_score = (self.sentiment_weight * sentiment_score +
                      self.image_confidence_weight * image_conf_score +
                      self.relevance_weight * relevance_score)
        
        return {
            'final_score': final_score,
            'sentiment_score': sentiment_score,
            'image_confidence_score': image_conf_score,
            'relevance_score': relevance_score
        }

if __name__ == "__main__":
    # Test
    fusion = FusionLayer()
    
    # Mock data
    sentiment_scores = {'negative': 0.1, 'neutral': 0.2, 'positive': 0.7}
    image_logits = torch.tensor([[2.0, 0.5, 1.0]])
    image_embedding = torch.rand(1, 768)
    text_embedding = torch.rand(1, 384)
    
    result = fusion.fuse(sentiment_scores, image_logits, image_embedding, text_embedding)
    print(f"Fusion result: {result}")
