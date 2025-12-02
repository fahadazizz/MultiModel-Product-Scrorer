from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image

class ImageClassifier:
    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, image):
        """
        Predicts the class of the image.
        Args:
            image: PIL Image or list of PIL Images
        Returns:
            logits: Model output logits
            predicted_class_idx: Index of the predicted class
            predicted_label: Label of the predicted class
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class_idx]
        
        return logits, predicted_class_idx, predicted_label

    def get_embeddings(self, image):
        """
        Extracts embeddings from the model (last hidden state before classification head).
        Note: ViTForImageClassification output usually contains hidden_states if configured, 
        but by default it returns logits. We might need to hook or use ViTModel if we want raw embeddings.
        However, for this task, the user mentioned "embeddings(classification) of image".
        Often the [CLS] token embedding is used.
        """
        # To get embeddings, we can use the hidden states.
        # We need to enable output_hidden_states=True when calling the model, or use ViTModel.
        # Let's re-init with output_hidden_states=True if we want to be safe, 
        # or just use the logits as a high-level representation if that's what's intended.
        # But usually "embeddings" implies the vector before the final layer.
        
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # The last hidden state is at index -1. Shape: (batch_size, sequence_length, hidden_size)
        # For ViT, the first token is [CLS] which represents the image.
        last_hidden_state = outputs.hidden_states[-1]
        cls_embedding = last_hidden_state[:, 0, :]
        
        return cls_embedding

if __name__ == "__main__":
    # Test
    classifier = ImageClassifier()
    img = Image.new('RGB', (224, 224), color='red')
    logits, idx, label = classifier.predict(img)
    print(f"Predicted: {label}")
    emb = classifier.get_embeddings(img)
    print(f"Embedding shape: {emb.shape}")
