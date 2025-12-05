from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
import torch
from PIL import Image

class ImageClassifier:
    def __init__(self, load_local_path="models/finetuned_vit_fahad"):
        self.processor = ViTImageProcessor.from_pretrained(load_local_path)
        self.model = ViTForImageClassification.from_pretrained(load_local_path, use_safetensors=True)
        self.vitModel = ViTModel.from_pretrained(load_local_path, use_safetensors=True)
        self.model.eval()
        self.vitModel.eval()

    def predict(self, image):
        """
        Predicts the class of the image.
        Args:
            image: PIL Image or list of PIL Images
        Returns:
            logits: Model output logits
            predicted_class: Index of the predicted class
            predicted_label: Label of the predicted class
        """
        inputs = self.processor(images=image, return_tensors="pt", padding = True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class]

        
        
        return logits, predicted_class, predicted_label

    def get_embeddings(self, image):
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        
        # Get features from ViTModel with pooling
        with torch.no_grad():
            outputs = self.vitModel(**inputs, output_hidden_states=True, add_pooling_layer=True)
        
        # Use the pooler_output as the CLS embedding
        cls_embedding = outputs.pooler_output  # shape: (batch_size, hidden_size)
        # print(cls_embedding.shape)
        
        return cls_embedding


