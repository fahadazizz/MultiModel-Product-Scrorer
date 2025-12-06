from PIL import Image
import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from product_review_analyzer import ProductReviewAnalyzer
import io
import uvicorn
import json

app = FastAPI(title="Multimodel Product Review Analyzer")

analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    analyzer = ProductReviewAnalyzer(finetuned_sentiment_path="models/trained/finetuned_roberta_fahad", 
    finetuned_vit_path="models/trained/finetuned_vit_fahad", 
    fusion_model_path="models/trained/fusionMLP_model.pth")

class TextAnalysisRequest(BaseModel):
    text: str

class ReviewRequest(BaseModel):
    reviews: List[str]

@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded product image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        logits, _, label = analyzer.image_classifier.predict(image)
        # Get max confidence score
        confidences = torch.softmax(logits, dim=1)
        max_score = torch.max(confidences).item()
        
        if max_score >= 0.5:
            return {
                "label": label,
                "score": max_score 
            }
        else:
            return {
                "label": "Please enter a valid image",
                "score": 0 
            }
        # return {
        #     "label": label,
        #     "score": max_score 
        # }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text")
async def analyze_text(request: TextAnalysisRequest):
    """Analyze sentiment of a review text."""
    try:
        scores, label = analyzer.sentiment_analyzer.analyze(request.text)
        # Return only the score for the predicted label
        label_score = scores[label]
        
        return {
            "label": label,
            "score": label_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    reviews: str = Form(...) 
):
    """
    Generate recommendation score from image and reviews.
    reviews: Can be a single string or a JSON list of strings.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result = analyzer.analyze(image, reviews)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
