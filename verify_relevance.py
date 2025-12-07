import os
import sys
from product_review_analyzer import ProductReviewAnalyzer
from PIL import Image
import torch

# Ensure we can import modules
sys.path.append(os.getcwd())

def verify_relevance():
    print("Initializing Analyzer...")
    analyzer = ProductReviewAnalyzer(
        finetuned_sentiment_path="models/trained/finetuned_roberta_fahad",
        finetuned_vit_path="models/trained/finetuned_vit_fahad",
        fusion_model_path="models/trained/fusion_multimodelMLP.pth" 
    )

    # Use a dummy image if file not found, or create one
    img_path = "dataset/multimodel_images/0.0_0.jpg"
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. Creating dummy white image.")
        img = Image.new('RGB', (224, 224), color='white')
    else:
        img = Image.open(img_path).convert('RGB')

    review_text = "This product is amazing and matches the description perfectly."

    print("\n--- Running Analysis ---")
    result = analyzer.analyze(img, review_text)

    print("\n--- Result ---")
    print(f"Final Score: {result['final_score']}")
    
    if 'relevance' in result:
        print(f"✅ Relevance Key Found")
        print(f"Relevance Score: {result['relevance']['score']}")
        print(f"Is Relevant: {result['relevance']['is_relevant']}")
    else:
        print(f"❌ Relevance Key Missing!")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_relevance()
