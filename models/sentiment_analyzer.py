from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

class SentimentAnalyzer:
    def __init__(self,load_local_path="models/finetuned_roberta_fahad"):
        self.tokenizer = AutoTokenizer.from_pretrained(load_local_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_local_path, use_safetensors=True)
    
        self.model.eval()
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
        Extract text embeddings from RoBERTa's CLS token (pooler_output).
        Args:
            text: String or list of strings
        Returns:
            embeddings: Tensor of shape (batch_size, hidden_size) or (hidden_size,) for single text
        """
        if isinstance(text, str):
            text = [text]
        
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model.roberta(**encoded_input)
        
        # RoBERTa pooler_output is the CLS token representation
        # If pooler is not available, we can use last_hidden_state[:, 0, :]
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            embeddings = output.pooler_output
        else:
            embeddings = output.last_hidden_state[:, 0, :]
        
        return embeddings
