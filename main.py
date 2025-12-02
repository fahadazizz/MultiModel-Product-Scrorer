from models.image_classifier import ImageClassifier
from models.text_embedder import TextEmbedder
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
        
        print("Loading Text Embedder...")
        self.text_embedder = TextEmbedder()
        
        print("Loading Sentiment Analyzer...")
        self.sentiment_analyzer = SentimentAnalyzer(load_local_path=finetuned_sentiment_path)
        
        print("Initializing Fusion Layer...")
        self.fusion_layer = FusionLayer()
        
        print("Analyzer ready!")

    def analyze(self, image_path, review_text):
        """
        Analyze a product image and review to generate recommendation score.
        
        Args:
            image_path: Path to product image (str) or PIL Image
            review_text: Product review text (str)
        
        Returns:
            Dictionary containing final score and component scores
        """
        # Load image if path provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # 1. Image Classification
        print("Classifying image...")
        image_logits, image_class_idx, image_label = self.image_classifier.predict(image)
        image_embedding = self.image_classifier.get_embeddings(image)
        
        # 2. Text Embedding
        print("Computing text embeddings...")
        text_embedding = self.text_embedder.get_embeddings([review_text])
        
        # 3. Sentiment Analysis
        print("Analyzing sentiment...")
        sentiment_scores, sentiment_label = self.sentiment_analyzer.analyze(review_text)
        
        # 4. Fusion
        print("Fusing scores...")
        fusion_result = self.fusion_layer.fuse(
            sentiment_scores=sentiment_scores,
            image_logits=image_logits,
            image_embedding=image_embedding,
            text_embedding=text_embedding
        )
        
        # Compile results
        result = {
            'final_score': fusion_result['final_score'],
            'recommendation': self._get_recommendation(fusion_result['final_score']),
            'components': {
                'sentiment': {
                    'label': sentiment_label,
                    'scores': sentiment_scores,
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
    # Example usage
    analyzer = ProductReviewAnalyzer()
    
    # Test with mock data
    result = analyzer.analyze(
        image_path="dataset/images/product_1.jpg",
        review_text="I absolutely love this product! It works perfectly and looks great."
    )
    
    print("\n" + "="*50)
    print("ANALYSIS RESULT")
    print("="*50)
    print(f"Final Score: {result['final_score']:.3f}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"\nSentiment: {result['components']['sentiment']['label']} ({result['components']['sentiment']['normalized_score']:.3f})")
    print(f"Image: {result['components']['image']['label']} ({result['components']['image']['confidence_score']:.3f})")
    print(f"Relevance: {result['components']['relevance']['score']:.3f}")
    print("="*50)
