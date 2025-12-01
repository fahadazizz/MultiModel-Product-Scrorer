import requests
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

imageURL = "https://m.media-amazon.com/images/I/715ohhkrQVL._AC_SY300_SX300_QL70_FMwebp_.jpg"
image = Image.open(url = requests.get(imageURL, stream=True).raw)
review = """
So far, I have used the 8" and 10" and I was impressed. Heat distribution was even and also retained it even when I added food.
The weight was heavier than I expected. Just out of curiosity, I weighed the 12" with the lid and it was over 5 lbs. That's almost as heavy as my 10" cast iron frying pan. That explains its heat retention.
A word of caution. Don't use high heat. 
"""

input = processor(text = review, images = image, return_tensors="pt", padding=True)
output = model(**input)

similarityScore = output.logits_per_image
print("The similarity score is: ", similarityScore)