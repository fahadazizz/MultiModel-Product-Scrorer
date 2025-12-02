# Multimodel Product Review Analyzer

A production-ready system that analyzes product reviews using multimodel AI to generate intelligent recommendation scores by combining image analysis, sentiment analysis, and semantic text embeddings.

## Features

- **Dual-Model Architecture**: Vision Transformer (ViT) for images + Fine-tuned RoBERTa for text
- **Unified Text Processing**: Single RoBERTa model handles both sentiment analysis and text embeddings
- **LoRA Fine-Tuning**: Parameter-efficient fine-tuning (only 0.94% trainable parameters)
- **Advanced Text Preprocessing**: Lemmatization, stopword removal, punctuation cleaning
- **Weighted Fusion Layer**: Combines sentiment (40%), image confidence (30%), and relevance (30%)
- **Comprehensive Evaluation**: F1, accuracy, and precision metrics

## Architecture

```
┌─────────────────┐         ┌──────────────────────┐
│  Product Image  │         │   Review Text        │
└────────┬────────┘         └──────────┬───────────┘
         │                             │
         ▼                             ▼
┌─────────────────┐         ┌──────────────────────┐
│  ViT Classifier │         │  RoBERTa (Fine-tuned)│
│  (Image Emb +   │         │  • Sentiment (head)  │
│   Classification)│         │  • Embeddings (CLS)  │
└────────┬────────┘         └──────────┬───────────┘
         │                             │
         │                             │
         └─────────────┬───────────────┘
                       ▼
              ┌────────────────┐
              │  Fusion Layer  │
              │  (Weighted Sum)│
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │ Recommendation │
              │     Score      │
              └────────────────┘
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from main import ProductReviewAnalyzer

# Initialize with fine-tuned model
analyzer = ProductReviewAnalyzer(
    finetuned_sentiment_path="models/finetuned_roberta"
)

# Analyze a product
result = analyzer.analyze(
    image_path="path/to/product.jpg",
    review_text="This product is amazing! Highly recommend."
)

print(f"Score: {result['final_score']:.3f}")
print(f"Recommendation: {result['recommendation']}")
print(f"Sentiment: {result['components']['sentiment']['label']}")
```

### Fine-Tuning

```bash
# Fine-tune RoBERTa on your dataset
python training/finetune_sentiment.py
```

The script includes:

- Text preprocessing (lowercasing, lemmatization, stopword removal)
- LoRA configuration for efficient fine-tuning
- Evaluation metrics (F1, accuracy, precision)

### Verification

```bash
# Run end-to-end tests
python verify.py
```

## Models Used

| Model                                              | Purpose                              | Source      |
| -------------------------------------------------- | ------------------------------------ | ----------- |
| `google/vit-base-patch16-224`                      | Image classification & embeddings    | HuggingFace |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment analysis & text embeddings | HuggingFace |

## Fine-Tuning Results

Trained on 200 samples (CPU-optimized):

- **Accuracy**: 75.0%
- **F1 Score**: 75.0%
- **Precision**: 75.1%
- **Trainable Parameters**: 1.18M / 125.8M (0.94%)

## Example Output

```
Final Score: 0.840
Recommendation: Highly Recommended

Component Scores:
  - Sentiment: positive (0.985)
  - Image Confidence: 0.994
  - Relevance: 0.493
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (for LoRA)
- scikit-learn
- NLTK
- Pillow

See `requirements.txt` for full list.

## Key Design Decisions

### Why RoBERTa for Both Tasks?

Instead of using separate models (e.g., MiniLM for embeddings), we use **one RoBERTa model** for:

1. **Sentiment Classification** (via classification head)
2. **Text Embeddings** (via CLS pooler_output)

Benefits:

- Reduced memory footprint
- Semantic consistency between tasks
- Fine-tuning improves both simultaneously
- Simpler deployment

### Fusion Layer Weights

```python
weights = {
    'sentiment': 0.4,        # Primary signal
    'image_confidence': 0.3, # Product quality indicator
    'relevance': 0.3         # Image-text alignment
}
```

Adjustable based on your use case.
