"""
Verification script for the Multimodel Product Review Analyzer.
Tests the complete pipeline with sample data.
"""

from main import ProductReviewAnalyzer
import os

def verify_analyzer():
    print("="*70)
    print("MULTIMODEL PRODUCT REVIEW ANALYZER - VERIFICATION")
    print("="*70)
    
    # Initialize analyzer (without fine-tuned model for now)
    print("\n[1/3] Initializing analyzer...")
    analyzer = ProductReviewAnalyzer()
    
    # Test cases
    test_cases = [
        {
            "name": "Positive Review",
            "image": "dataset/images/product_1.jpg",
            "review": "I absolutely love this product! It works perfectly and looks great."
        },
        {
            "name": "Negative Review",
            "image": "dataset/images/product_2.jpg",
            "review": "The quality is terrible. It broke after one use. Do not buy."
        },
        {
            "name": "Neutral Review",
            "image": "dataset/images/product_3.jpg",
            "review": "It's okay, not the best but does the job for the price."
        },
        {
            "name": "Highly Positive Review",
            "image": "dataset/images/product_4.jpg",
            "review": "Amazing experience, highly recommended!"
        },
        {
            "name": "Very Negative Review",
            "image": "dataset/images/product_5.jpg",
            "review": "Waste of money. Very disappointed."
        }
    ]
    
    print(f"\n[2/3] Running {len(test_cases)} test cases...\n")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}/{len(test_cases)}: {test_case['name']}")
        print('='*70)
        
        if not os.path.exists(test_case['image']):
            print(f"⚠️  Image not found: {test_case['image']}")
            continue
        
        try:
            result = analyzer.analyze(
                image_path=test_case['image'],
                review_text=test_case['review']
            )
            
            results.append({
                'test_case': test_case['name'],
                'result': result
            })
            
            print(f"\n📊 RESULTS:")
            print(f"   Review: \"{test_case['review']}\"")
            print(f"\n   ✅ Final Score: {result['final_score']:.3f}")
            print(f"   ✅ Recommendation: {result['recommendation']}")
            print(f"\n   Component Scores:")
            print(f"      - Sentiment: {result['components']['sentiment']['label']} "
                  f"({result['components']['sentiment']['normalized_score']:.3f})")
            print(f"      - Image Confidence: {result['components']['image']['confidence_score']:.3f}")
            print(f"      - Relevance: {result['components']['relevance']['score']:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n\n{'='*70}")
    print("[3/3] VERIFICATION SUMMARY")
    print('='*70)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(test_cases) - len(results)}")
    
    if len(results) == len(test_cases):
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n⚠️  {len(test_cases) - len(results)} test(s) failed")
    
    print('='*70)
    
    return results

if __name__ == "__main__":
    verify_analyzer()
