from models.image_classifier import ImageClassifier
from models.sentiment_analyzer import SentimentAnalyzer
from fusion.fusion_layer import FusionLayer
from PIL import Image

class ProductReviewAnalyzer:
    def __init__(self, finetuned_sentiment_path=None):
        """
        Initialize the multimodel product review analyzer.
        
        Args:
            finetuned_sentiment_path: Path to fine-tuned sentiment model (optional)
        """
        print("Loading Image Classifier...")
        self.image_classifier = ImageClassifier()
        
        print("Loading Sentiment Analyzer (with text embeddings)...")
        self.sentiment_analyzer = SentimentAnalyzer(load_local_path=finetuned_sentiment_path)
        
        print("Initializing Fusion Layer...")
        self.fusion_layer = FusionLayer()
        
        print("Analyzer ready!")

    def analyze(self, image_path, reviews):
        """
        Analyze a product image and multiple reviews to generate recommendation score.
        
        Args:
            image_path: Path to product image (str) or PIL Image
            reviews: List of product review texts (list of str) or single review (str)
        
        Returns:
            Dictionary containing final score and component scores
        """
        # Load image if path provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Ensure reviews is a list
        if isinstance(reviews, str):
            reviews = [reviews]
        
        # 1. Image Classification
        print("Classifying image...")
        image_logits, image_class_idx, image_label = self.image_classifier.predict(image)
        image_embedding = self.image_classifier.get_embeddings(image)
        
        # 2. Multi-Review Analysis & Aggregation
        print(f"Analyzing {len(reviews)} reviews...")
        
        all_sentiment_scores = []
        all_text_embeddings = []
        
        for review_text in reviews:
            # Sentiment Analysis
            scores, _ = self.sentiment_analyzer.analyze(review_text)
            all_sentiment_scores.append(scores)
            
            # Text Embedding
            embedding = self.sentiment_analyzer.get_embeddings(review_text)
            all_text_embeddings.append(embedding)
            
        # Aggregate Sentiment Scores (Mean probability per class)
        avg_sentiment_scores = {
            'negative': sum(s['negative'] for s in all_sentiment_scores) / len(reviews),
            'neutral': sum(s['neutral'] for s in all_sentiment_scores) / len(reviews),
            'positive': sum(s['positive'] for s in all_sentiment_scores) / len(reviews)
        }
        
        # Determine aggregated label
        avg_sentiment_label = max(avg_sentiment_scores, key=avg_sentiment_scores.get)
        
        # Aggregate Text Embeddings (Mean vector)
        if len(all_text_embeddings) > 0:
            import torch
            stacked_embeddings = torch.stack(all_text_embeddings)
            avg_text_embedding = torch.mean(stacked_embeddings, dim=0)
        else:
            avg_text_embedding = None # Should not happen if reviews list is not empty
            
        # 3. Fusion with Aggregated Values
        print("Fusing scores...")
        fusion_result = self.fusion_layer.fuse(
            sentiment_scores=avg_sentiment_scores,
            image_logits=image_logits,
            image_embedding=image_embedding,
            text_embedding=avg_text_embedding
        )
        
        # Compile results
        result = {
            'final_score': fusion_result['final_score'],
            'recommendation': self._get_recommendation(fusion_result['final_score']),
            'components': {
                'sentiment': {
                    'label': avg_sentiment_label,
                    'scores': avg_sentiment_scores,
                    'normalized_score': fusion_result['sentiment_score']
                },
                'image': {
                    'label': image_label,
                    'class_idx': image_class_idx,
                    'confidence_score': fusion_result['image_confidence_score']
                },
                'relevance': {
                    'score': fusion_result['relevance_score']
                }
            }
        }
        
        return result

    def _get_recommendation(self, score):
        """Convert score to recommendation category."""
        if score >= 0.7:
            return "Highly Recommended"
        elif score >= 0.5:
            return "Recommended"
        elif score >= 0.3:
            return "Neutral"
        else:
            return "Not Recommended"

if __name__ == "__main__":
    analyzer = ProductReviewAnalyzer()
    # Test with mock data
    result = analyzer.analyze(
        image_path="dataset/images/tshirt.jpg",
        reviews=[
            """I thought it might be to big for me
Im 5'2 female and 160 pounds and it fits perfect. Its not super long and its not tight at all.
The t shirt material is the softer kind. Which makes it cooler in summer.
I am a gamer and yes its true.
I have had to stop playing my game to go out to dinner with family.""",
"The printing on the shirt was very poor quality. It was packaged in clear plastic bag that was torn open. I cannot recommend this product.",
"""
The picture is misleading, I thought it was a fitted style, but now realize they just take a stock picture and change the graphic on the shirt. 
Some reviews said it runs small, so I order a men's XL, but it was basically a square. Runs plenty wide, but short. Returned and ordered from etsy.
"""
        ]
    )
    print(result)