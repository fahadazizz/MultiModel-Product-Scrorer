from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image

import os
import torch.nn.functional as F

class ImageClassifier:
    def __init__(self, model_path=None, model_name='google/vit-base-patch16-224'):
        # Use simple logic: if path is provided, use it; otherwise use default name
        path_to_use = model_path if model_path else model_name
        
        # Ensure absolute path for robustness
        if os.path.exists(path_to_use):
            path_to_use = os.path.abspath(path_to_use)
            
        print(f"Loading Image Model from: {path_to_use}")
        
        self.processor = ViTImageProcessor.from_pretrained(path_to_use)
        self.model = ViTForImageClassification.from_pretrained(path_to_use)
        self.model.eval()

    def predict(self, image):
        """
        Predicts the class of the image.
        Args:
            image: PIL Image or list of PIL Images
        Returns:
            logits: Model output logits
            predicted_class: Index of the predicted class
            predicted_label: Label of the predicted class (or warning if low confidence)
        """
        inputs = self.processor(images=image, return_tensors="pt", padding = True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        max_score = torch.max(probs).item()
        
        predicted_class = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class]
        
        # Threshold Check
        if max_score < 0.4:
            predicted_label = "please add correct image"
        
        return logits, predicted_class, predicted_label

    def get_embeddings(self, image):
        # To get embeddings, we can use the hidden states.
        # We need to enable output_hidden_states=True when calling the model, or use ViTModel.
        # Usually "embeddings" implies the vector before the final layer.
        
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # The last hidden state is at index -1. Shape: (batch_size, sequence_length, hidden_size)
        # For ViT, the first token is [CLS] which represents the image.
        last_hidden_state = outputs.hidden_states[-1]
        cls_embedding = last_hidden_state[:, 0, :]
        
        
        return cls_embedding

