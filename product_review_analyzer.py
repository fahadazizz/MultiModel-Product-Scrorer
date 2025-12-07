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
                 fusion_model_path="models/trained/fusion_multimodelMLP.pth"):
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
        
        model.eval() # Ensure model is in eval mode for inference
        return model

    def _extract_image_embedding(self, image):
        """Extract ViT embedding for the image (Sequence)"""
        embedding = self.image_classifier.get_sequence_embeddings(image)
        print("Image embedding extracted (Sequence)")
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
        image_score = torch.max(confidences).item()

        if image_score < 0.5:
            return {
                "score": 0.0,
                "recommendation": "NILL",
                "components": {
                    "sentiment": {
                        "label": "Pleasa Provide correct image",
                        "score": {"positive": 0, "negative": 0}
                    },
                    "visual": {
                        "label": "NILL",
                        "score": 0.0
                    },
                    "fusion": {
                        "relevance_similarity": 0.0,
                        "normalized_score": 0.0
                    }
                }
            }

        # Get sentiment analysis for context
        print("Getting sentiment analysis...")
        sentiment_scores, sentiment_label = self.sentiment_analyzer.analyze(reviews)
        print("sentiment scores", sentiment_scores)
        print("sentiment label", sentiment_label)
        sentiment_score = sentiment_scores # Added sentiment_score for consistency with new output format


        print("Extracting image embedding...")
        image_embedding = self._extract_image_embedding(image)
        
        print("Extracting text embedding...")
        text_embedding = self._extract_text_embedding(reviews)
        
        # 1. Relevance Gating (New Feature)
        # Compute cosine similarity between image and text
        with torch.no_grad():
            similarity = self.fusion_model.compute_similarity(image_embedding, text_embedding).item()
            
        print(f"Relevance Similarity: {similarity:.4f}")
        
        # Threshold T = 0.2 (Can be tuned)
        # Cosine similarity range [-1, 1]. Unrelated/Random is usually ~0.
        if similarity < 0.3:
             return {
                "score": 0.0,
                "recommendation": "Irrelevant image and review",
                "components": {
                    "sentiment": {"label": sentiment_label, "score": sentiment_score},
                    "visual": {"label": image_label, "score": image_score},
                    "fusion": {"relevance_similarity": similarity}
                }
             }

        # 2. Get recommendation score from trained fusion model
        print("Getting recommendation score from trained fusion model...")
        with torch.no_grad():
            # Pass sequence (B, 197, 768) and CLS (B, 768)
            score = self.fusion_model(image_embedding, text_embedding) 
            
        final_score_raw = score.item()
        
        # 3. Denormalize score (0-1 -> 1-10)
        normalized_score = final_score_raw
        final_score = 1 + final_score_raw * 9
        
        # Clamp to 1-10
        final_score = max(1.0, min(10.0, final_score))
        
        recommendation = self._get_recommendation(final_score)
        
        # Compile results
        result = {
            "score": final_score,
            "recommendation": recommendation,
            "components": {
                "sentiment": {
                    "label": sentiment_label, 
                    "score": sentiment_score
                },
                "visual": {     
                    "label": image_label, 
                    "score": image_score 
                },
                "fusion": {
                    "normalized_score": normalized_score,
                    "relevance_similarity": similarity
                }
            }
        }
        
        return result

    def _get_recommendation(self, score):
        """Convert score to recommendation category."""
        if score >= 5:
            return "Recommended"
        elif score >= 3:
            return "Neutral"
        else:
            return "Not Recommended"
