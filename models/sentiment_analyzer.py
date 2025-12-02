from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

class SentimentAnalyzer:
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest', load_local_path=None):
        if load_local_path:
            self.tokenizer = AutoTokenizer.from_pretrained(load_local_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_local_path, use_safetensors=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
        
        self.model.eval()
        self.labels = ['negative', 'neutral', 'positive']

    def analyze(self, text):
        """
        Analyze sentiment of the text.
        Args:
            text: String
        Returns:
            scores: Dictionary of label -> score
            predicted_label: String
        """
        encoded_input = self.tokenizer(text, return_tensors='pt')
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

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    text = "This product is amazing!"
    scores, label = analyzer.analyze(text)
    print(f"Text: {text}")
    print(f"Label: {label}")
    print(f"Scores: {scores}")
