from models.image_classifier import ImageClassifier
from models.sentiment_analyzer import SentimentAnalyzer
from models.fusion_mlp import MultimodalFusionWithAttention
from PIL import Image
import os
import torch


class ProductReviewAnalyzer:
    def __init__(
        self, 
        finetuned_sentiment_path="models/trained/finetuned_roberta_fahad", 
        finetuned_vit_path="models/trained/finetuned_vit_fahad",
        fusion_model_path="models/trained/CrossAttentMLP.pth"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading Image Classifier...")
        self.image_classifier = ImageClassifier(load_local_path=finetuned_vit_path)
        
        print("Loading Sentiment Analyzer...")
        self.sentiment_analyzer = SentimentAnalyzer(load_local_path=finetuned_sentiment_path)
        
        print("Loading Cross-Modal Attention Fusion Model...")
        self.fusion_model = self._load_fusion_model(fusion_model_path)
        
        print("Analyzer ready!")

    def _load_fusion_model(self, model_path):
        model = MultimodalFusionWithAttention(
            embed_dim=768,
            num_heads=8,
            mlp_hidden_dims=(512, 128, 32),
            dropout=0.2,
            attention_dropout=0.1
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded fusion model from {model_path}")
            if 'rmse' in checkpoint:
                print(f"  Model RMSE: {checkpoint['rmse']:.4f}")
        else:
            print(f"Warning: Fusion model not found at {model_path}. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model

    def _extract_image_sequence(self, image):
        sequence = self.image_classifier.get_sequence_embeddings(image)
        return sequence.to(self.device)

    def _extract_text_sequence(self, reviews):
        sequence, attention_mask = self.sentiment_analyzer.get_sequence_embeddings(reviews)
        return sequence.to(self.device), attention_mask.to(self.device)

    def analyze(self, image_path, reviews):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        print("Getting image classification...")
        logits, _, image_label = self.image_classifier.predict(image)
        confidences = torch.softmax(logits, dim=1)
        max_score = torch.max(confidences).item()

        if max_score < 0.5:
            return {
                "final_score": 0,
                "recommendation": "INVALID",
                "components": {
                    "sentiment": {
                        "label": "Please provide correct image",
                        "scores": {"positive": 0, "negative": 0, "neutral": 0}
                    },
                    "image": {
                        "label": "INVALID",
                        "confidence_score": 0
                    },
                    "relevance": {
                        "score": 0,
                        "is_relevant": False,
                        "alignment_score": 0,
                        "shift_score": 0
                    }
                }
            }

        print("Getting sentiment analysis...")
        sentiment_scores, sentiment_label = self.sentiment_analyzer.analyze(reviews)

        print("Extracting image sequence embeddings...")
        image_seq = self._extract_image_sequence(image)
        
        print("Extracting text sequence embeddings...")
        text_seq, text_mask = self._extract_text_sequence(reviews)
        
        print("Computing cross-modal attention and relevance...")
        with torch.no_grad():
            # Get score AND relevance details
            normalized_score, relevance_info = self.fusion_model(
                image_seq=image_seq,
                text_seq=text_seq,
                text_attention_mask=text_mask,
                return_relevance=True
            )
            
            # Extract relevance metrics
            relevance_score = relevance_info['relevance_score'].cpu().item()
            alignment_score = relevance_info['alignment_score'].cpu().item()
            shift_score = relevance_info['shift_score'].cpu().item()
            is_relevant = relevance_info['is_relevant'].cpu().item()
            
            converted_score = 1 + normalized_score.cpu().item() * 9
            final_score = max(1.0, min(10.0, converted_score))
            
        # Log relevance details
        print(f"  Alignment Score: {alignment_score:.4f}")
        print(f"  Shift Score: {shift_score:.4f}")
        print(f"  Relevance Score: {relevance_score:.4f} ({'✓ Relevant' if is_relevant else '✗ Not Relevant'})")

        result = {
            'final_score': final_score,
            'recommendation': self._get_recommendation(final_score, is_relevant),
            'components': {
                'sentiment': {
                    'label': sentiment_label,
                    'scores': sentiment_scores,
                    'normalized_score': sentiment_scores.get(sentiment_label, 0.5)
                },
                'image': {
                    'label': image_label,
                    'confidence_score': max_score
                },
                'relevance': {
                    'score': relevance_score,
                    'is_relevant': bool(is_relevant),
                    'alignment_score': alignment_score,
                    'shift_score': shift_score,
                    'threshold': MultimodalFusionWithAttention.RELEVANCE_THRESHOLD
                }
            }
        }
        
        print(f"Final Score: {final_score:.2f} - {result['recommendation']}")
        return result

    def _get_recommendation(self, score, is_relevant=True):
        # If not relevant (below threshold), warn the user
        if not is_relevant:
            return "Low Relevance - Review may not match image"
        
        if score >= 3.5:
            return "Recommended"
        elif score >= 2:
            return "Neutral"
        else:
            return "Not Recommended"
