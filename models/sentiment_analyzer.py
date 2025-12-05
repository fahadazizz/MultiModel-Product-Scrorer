from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaModel
from scipy.special import softmax
import torch

class SentimentAnalyzer:
    def __init__(self,load_local_path="models/finetuned_roberta_fahad"):
        self.tokenizer = AutoTokenizer.from_pretrained(load_local_path)
        self.model = RobertaForSequenceClassification.from_pretrained(load_local_path, use_safetensors=True)
        self.roberta_model = RobertaModel.from_pretrained(load_local_path, use_safetensors=True)
    
        self.model.eval()
        self.roberta_model.eval()
        self.labels = ['negative', 'neutral', 'positive']

    def analyze(self, text):
        """
        Analyze sentiment of the text.
        Args:
            text: String or list of strings
        Returns:
            scores: Dictionary of label -> score
            predicted_label: String
        """
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        scores = output.logits[0].numpy()
        scores = softmax(scores)
        
        ranking = torch.argsort(torch.tensor(scores), descending=True)
        predicted_label = self.labels[ranking[0]]
        
        result_scores = {
            self.labels[0]: float(scores[0]),
            self.labels[1]: float(scores[1]),
            self.labels[2]: float(scores[2])
        }
        
        return result_scores, predicted_label
    
    def get_embeddings(self, text):
        """
        Extract text embeddings from RoBERTa's CLS token.
        Args:
            text: String or list of strings
        Returns:
            embeddings: Tensor of shape (batch_size, hidden_size) or (hidden_size,) for single text
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize input
        encoded_input = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            # Forward pass through RobertaModel (base model)
            output = self.roberta_model(**encoded_input)

        embedding = output.last_hidden_state[:, 0, :]

        return embedding
