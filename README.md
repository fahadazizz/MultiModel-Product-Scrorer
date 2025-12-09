# 🛍️ Multimodel Product Review Analyzer

A production-ready **multimodal AI system** that analyzes product images and customer reviews to generate intelligent recommendation scores using **Cross-Modal Attention**.

## ✨ Key Features

- **Cross-Modal Attention**: Text reviews attend to relevant image patches for context-aware fusion
- **Fine-tuned Backbones**: ViT for images + RoBERTa for text (both domain-adapted)
- **End-to-End Pipeline**: From raw image/text → recommendation score (1-10)
- **FastAPI Backend**: Production-ready REST endpoints
- **Streamlit Dashboard**: Interactive UI for analysis

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   FROZEN BACKBONES                       │
                    │  ┌─────────────────┐         ┌──────────────────────┐   │
                    │  │  Product Image  │         │    Review Text       │   │
                    │  └────────┬────────┘         └──────────┬───────────┘   │
                    │           │                             │               │
                    │           ▼                             ▼               │
                    │  ┌─────────────────┐         ┌──────────────────────┐   │
                    │  │   ViT-Base      │         │   RoBERTa-Base       │   │
                    │  │ (Fine-tuned)    │         │   (Fine-tuned)       │   │
                    │  └────────┬────────┘         └──────────┬───────────┘   │
                    │           │                             │               │
                    │           ▼                             ▼               │
                    │  Image Sequence               Text Sequence             │
                    │  (B, 197, 768)                (B, 128, 768)             │
                    └───────────┬─────────────────────────┬───────────────────┘
                                │                         │
                                └───────────┬─────────────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │              TRAINABLE FUSION                  │
                    │  ┌─────────────────────────────────────────┐  │
                    │  │        Cross-Modal Attention            │  │
                    │  │   Q = Text Tokens, K = V = Image Patches │  │
                    │  │         (8 attention heads)             │  │
                    │  └──────────────────┬──────────────────────┘  │
                    │                     │                         │
                    │                     ▼                         │
                    │  ┌─────────────────────────────────────────┐  │
                    │  │           Pooling + Concat              │  │
                    │  │  [Mean(Attended_Text), CLS(Image)]      │  │
                    │  │           Shape: (B, 1536)              │  │
                    │  └──────────────────┬──────────────────────┘  │
                    │                     │                         │
                    │                     ▼                         │
                    │  ┌─────────────────────────────────────────┐  │
                    │  │           MLP Score Head                │  │
                    │  │    1536 → 512 → 128 → 32 → 1 (σ)        │  │
                    │  └──────────────────┬──────────────────────┘  │
                    └─────────────────────┼─────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────┐
                              │ Recommendation     │
                              │ Score (1-10)       │
                              └────────────────────┘
```

### Why Cross-Modal Attention?

Instead of simple concatenation, our model uses **text-queries-image attention**:
- Text tokens can "look at" relevant image patches
- If review says "cracked screen", attention focuses on screen regions
- Results in context-aware, semantically meaningful fusion

---



## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the API Server

```bash
python3 main.py
```

API endpoints:
- `POST /classify-image` - Classify product image
- `POST /analyze-text` - Analyze review sentiment
- `POST /recommend` - Full multimodal recommendation

### Run the Dashboard

```bash
streamlit run dashboard.py
```

---

## 💻 Python Usage

```python
from product_review_analyzer import ProductReviewAnalyzer

# Initialize
analyzer = ProductReviewAnalyzer(
    fusion_model_path="models/trained/CrossAttentMLP.pth"
)

# Analyze
result = analyzer.analyze(
    image_path="path/to/product.jpg",
    reviews="Amazing product! The quality is excellent."
)

print(f"Score: {result['final_score']:.1f}/10")
print(f"Recommendation: {result['recommendation']}")
```

---

## 🎯 Training Pipeline

### 1. Fine-tune ViT (Image Classification)
```bash
python training/finetune_vit.py
```

### 2. Fine-tune RoBERTa (Sentiment Analysis)
```bash
python training/finetune_sentiment.py
```

### 3. Train Cross-Modal Fusion
```bash
python training/fusion_trainer.py
```

---

## 📊 Model Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **Fusion Model** | RMSE | 0.695 |
| **Fusion Model** | Parameters | 3.2M |
| **ViT** | Accuracy | ~85% |
| **RoBERTa** | F1 Score | ~75% |

## 🛠️ Tech Stack

- **PyTorch** - Deep learning framework
- **Transformers** - ViT and RoBERTa models
- **FastAPI** - REST API
- **Streamlit** - Dashboard UI
- **PEFT** - LoRA fine-tuning
