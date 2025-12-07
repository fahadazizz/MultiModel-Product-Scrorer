import sys
import os
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from product_review_analyzer import ProductReviewAnalyzer

def verify_gating():
    print("Initializing Analyzer...")
    analyzer = ProductReviewAnalyzer()
    
    # Path to a test image (ensure this exists)
    image_path = "dataset/multimodel_images/0.0_0.jpg"
    
    if not os.path.exists(image_path):
        print(f"Warning: Test image {image_path} not found. Utilizing a dummy image for testing not implemented yet.")
        # Create a dummy image if needed or just fail
        return

    # Case 1: Likely Irrelevant (Random Text vs Image)
    # Untrained projection heads -> random similarity -> likely low -> should be rejected or have low sim.
    print("\n--- Test Case 1: Review ---")
    review_text = "This is a random text about politics and economics. It has nothing to do with the image."
    result = analyzer.analyze(image_path, review_text)
    
    print("\nResult 1:")
    print(result)
    
    if "relevance_similarity" in result["components"]["fusion"]:
        sim = result["components"]["fusion"]["relevance_similarity"]
        print(f"✅ Relevance Similarity present: {sim}")
    else:
        print("❌ Relevance Similarity MISSING in output!")

    if "relevance_gating" in result["components"]["fusion"]:
         print(f"✅ Gating Triggered (sim < 0.2). Value: {result['components']['fusion']['relevance_gating']}")
    
    # Case 2: Check if threshold logic works
    # If sim was low, result['score'] should be 0.
    if result['score'] == 0.0 and result['recommendation'].startswith("Irrelevant"):
        print("✅ Gating Logic Verified (Score is 0 for low similarity).")
    elif result['score'] > 0:
        print(f"ℹ️ Similarity {sim} was >= 0.2, so scoring proceeded.")
        
if __name__ == "__main__":
    verify_gating()
