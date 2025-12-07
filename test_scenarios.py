import os
import torch
from product_review_analyzer import ProductReviewAnalyzer

def run_tests():
    print("Initializing Analyzer...")
    analyzer = ProductReviewAnalyzer()
    
    image_path = "dataset/images/tshirt/1.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return

    print(f"Testing with Image: {image_path}")

    # Test Scenarios
    scenarios = [
        {
            "name": "Short Relevant Review",
            "text": "This t-shirt fits perfectly and the material feels high quality.",
            "expected_relevance": "High"
        },
        {
            "name": "Long Relevant Review",
            "text": "I was skeptical about buying clothes online, but this t-shirt exceeded my expectations. The stitching is durable, the color hasn't faded after multiple washes, and it's incredibly soft. I highly recommend it for casual wear.",
            "expected_relevance": "High"
        },
        {
            "name": "Short Irrelevant Review",
            "text": "The coffee tastes burnt and bitter.",
            "expected_relevance": "Low"
        },
        {
            "name": "Long Irrelevant Review",
            "text": "I tried to install the graphics card drivers but the screen kept flickering. Support was unhelpful and I eventually had to return the laptop. Worst electronics purchase ever.",
            "expected_relevance": "Low"
        }
    ]

    print("\n--- Starting Comprehensive Tests ---")
    
    for i, test in enumerate(scenarios, 1):
        print(f"\nTest Case {i}: {test['name']}")
        print(f"Review Text: \"{test['text']}\"")
        
        try:
            # Run Analysis
            # analyze returns a single dictionary for the input
            result = analyzer.analyze(image_path, [test['text']])
            
            res = result
            
            fusion_comp = res["components"]["fusion"]
            sim_score = fusion_comp.get("relevance_similarity", 0.0)
            score = res.get("score", 0.0)
            
            print(f"  -> Similarity Score: {sim_score:.4f}")
            print(f"  -> Final Score: {score}")
            print(f"  -> Recommendation: {res['recommendation']}")
            
            # Determine Pass/Fail based on Logic Execution
            # (Note: Random untrained model might give unexpected Similarity values, so we judge the LOGIC)
            
            gating_threshold = 0.2
            is_gated = sim_score < gating_threshold
            
            if is_gated:
                print("  -> Logic: GATED (Irrelevant)")
                if score == 0.0:
                    print("  ✅ PASS: Score is 0.0 as expected for Gated review.")
                else:
                    print(f"  ❌ FAIL: Score should be 0.0 for Gated review, got {score}")
            else:
                print("  -> Logic: ALLOWED (Relevant)")
                if score > 0.0:
                    print("  ✅ PASS: Score generated (> 0.0).")
                else:
                    print("  ❌ FAIL: Score is 0.0 but it was allowed?")
            
            # Expectation Check (Informational only for untrained model)
            if test["expected_relevance"] == "Low" and not is_gated:
                 print("  ⚠️ Model Note: Irrelevant text was ALLOWED (Model needs training)")
            elif test["expected_relevance"] == "High" and is_gated:
                 print("  ⚠️ Model Note: Relevant text was GATED (Model needs training)")
                 
        except Exception as e:
            print(f"  ❌ FAIL: Exception occurred - {e}")

if __name__ == "__main__":
    run_tests()
