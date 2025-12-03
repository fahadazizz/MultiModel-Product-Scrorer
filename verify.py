from main import ProductReviewAnalyzer
import os

def verify_analyzer():
    print("="*70)
    print("MULTIMODEL PRODUCT REVIEW ANALYZER - VERIFICATION")
    print("="*70)
    
    # Initialize analyzer
    print("\n[1/3] Initializing analyzer...")
    print("Using base RoBERTa model")
    analyzer = ProductReviewAnalyzer()
    
    # Test cases
    test_cases = [
        {
            "name": "Positive Reviews (Multiple)",
            "image": "dataset/images/samsung.jpg",
            "reviews": [
                "I absolutely love this product! It works perfectly.",
                "Great design and features. Highly recommended."
            ]
        },
        {
            "name": "Mixed Reviews (Positive + Negative)",
            "image": "dataset/images/tshirt.jpg",
            "reviews": [
                "The material is soft and comfortable.",
                "However, the stitching started coming apart after one wash. Terrible quality."
            ]
        },
        {
            "name": "Neutral/Average Reviews",
            "image": "dataset/images/samsung.jpg",
            "reviews": [
                "It's okay, does the job.",
                "Not the best, not the worst. Average product."
            ]
        },
        {
            "name": "Single Review (Backward Compatibility)",
            "image": "dataset/images/tshirt.jpg",
            "reviews": "Amazing experience, highly recommended!"
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
                reviews=test_case['reviews']
            )
            
            results.append({
                'test_case': test_case['name'],
                'result': result
            })
            
            print(f"\n📊 RESULTS:")
            if isinstance(test_case['reviews'], list):
                print(f"   Reviews ({len(test_case['reviews'])}):")
                for r in test_case['reviews']:
                    print(f"     - \"{r}\"")
            else:
                print(f"   Review: \"{test_case['reviews']}\"")
            print(f"\n   Final Score: {result['final_score']:.3f}")
            print(f"   Recommendation: {result['recommendation']}")
            print(f"\n   Component Scores:")
            print(f"      - Sentiment: {result['components']['sentiment']['label']} "
                  f"({result['components']['sentiment']['normalized_score']:.3f})")
            print(f"      - Image Confidence: {result['components']['image']['confidence_score']:.3f}")
            print(f"      - Relevance: {result['components']['relevance']['score']:.3f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
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
