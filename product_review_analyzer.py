from models.image_classifier import ImageClassifier
from models.sentiment_analyzer import SentimentAnalyzer
from PIL import Image
import os
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor, RobertaModel, AutoTokenizer
from models.fusion_mlp import MultimodalFusion
import numpy as np

class ProductReviewAnalyzer:
    def __init__(self, 
                 finetuned_sentiment_path="models/trained/finetuned_roberta_fahad", 
                 finetuned_vit_path="models/trained/finetuned_vit_fahad",
                 fusion_model_path="models/trained/fusionMLP_model.pth"):
        """
        Initialize the multimodel product review analyzer with your new fusion model.
        
        Args:
            finetuned_sentiment_path: Path to fine-tuned sentiment model
            finetuned_vit_path: Path to fine-tuned ViT model  
            fusion_model_path: Path to your trained fusion model
        """
        print("Loading Image Classifier...")
        self.image_classifier = ImageClassifier(load_local_path=finetuned_vit_path)
        
        print("Loading Sentiment Analyzer...")
        self.sentiment_analyzer = SentimentAnalyzer(load_local_path=finetuned_sentiment_path)
        
        print("Loading Trained Fusion Model...")
        self.fusion_model = self._load_fusion_model(fusion_model_path)
        
        print("Analyzer ready!")

    def _load_fusion_model(self, model_path):
        """Load your trained fusion model"""
        model = MultimodalFusion()
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            #model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded fusion model from {model_path}")
        else:
            print(f"Warning: Fusion model not found at {model_path}. Using untrained model.")
        
        return model

    def _extract_image_embedding(self, image):
        """Extract ViT embedding for the image"""
        embedding = self.image_classifier.get_embeddings(image)
        print("Image embeding extracted")
        return embedding

    def _extract_text_embedding(self, reviews):
        embedding = self.sentiment_analyzer.get_embeddings(reviews)
        print("Text embeding extracted")
        return embedding

    def analyze(self, image_path, reviews):
        """
        Analyze a product image and multiple reviews using your trained fusion model.
        
        Args:
            image_path: Path to product image (str) or PIL Image
            reviews: List of product review texts (list of str) or single review (str)
        
        Returns:
            Dictionary containing final score and component analysis
        """
        # Load image if path provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        

        # Get image classification for context
        print("Getting image classification...")
        logits, _, image_label = self.image_classifier.predict(image)
        # Get max confidence score
        confidences = torch.softmax(logits, dim=1)
        max_score = torch.max(confidences).item()

        if max_score < 0.5:
            return {
                "final_score": 0,
                "recommendation": "NILL",
                "components": {
                    "sentiment": {
                        "label": "NILL",
                        "scores": {"positive": 0, "negative": 0}
                    },
                    "image": {
                        "label": "NILL",
                        "confidence_score": 0
                    }
                }
            }

        # Get sentiment analysis for context
        print("Getting sentiment analysis...")
        sentiment_scores, sentiment_label = self.sentiment_analyzer.analyze(reviews)



        print("Extracting image embedding...")
        image_embedding = self._extract_image_embedding(image)
        
        print("Extracting text embedding...")
        text_embedding = self._extract_text_embedding(reviews)
        
        print("Getting recommendation score from trained fusion model...")
        with torch.no_grad():
            # Get prediction from your trained fusion model
            
            normalized_score = self.fusion_model(image_embedding, text_embedding)
            
            # Convert from 0-1 normalized scale back to 1-10 scale
            final_score = 1 + normalized_score.cpu().item() * 9
            
            # Clamp to valid range
            final_score = max(1.0, min(10.0, final_score))
            

        # Compile results
        result = {
            'final_score': round(final_score, 2),
            'recommendation': self._get_recommendation(final_score),
            'components': {
                'sentiment': {
                    'label': sentiment_label,
                    'scores': sentiment_scores
                },
                'image': {
                    'label': image_label,
                    'confidence': confidences
                }
            }
        }
        
        return result

    def _get_recommendation(self, score):
        """Convert score to recommendation category."""
        if score >= 8.0:
            return "Highly Recommended"
        elif score >= 6.0:
            return "Recommended"
        elif score >= 3.0:
            return "Neutral"
        else:
            return "Not Recommended"
