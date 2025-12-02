import pandas as pd
import os
from PIL import Image, ImageDraw

def create_mock_dataset(dataset_dir="dataset"):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    data = [
        {
            "id": 1,
            "image_path": os.path.join(images_dir, "product_1.jpg"),
            "review_text": "I absolutely love this product! It works perfectly and looks great.",
            "sentiment": "positive",
            "image_color": "green"
        },
        {
            "id": 2,
            "image_path": os.path.join(images_dir, "product_2.jpg"),
            "review_text": "The quality is terrible. It broke after one use. Do not buy.",
            "sentiment": "negative",
            "image_color": "red"
        },
        {
            "id": 3,
            "image_path": os.path.join(images_dir, "product_3.jpg"),
            "review_text": "It's okay, not the best but does the job for the price.",
            "sentiment": "neutral",
            "image_color": "grey"
        },
        {
            "id": 4,
            "image_path": os.path.join(images_dir, "product_4.jpg"),
            "review_text": "Amazing experience, highly recommended!",
            "sentiment": "positive",
            "image_color": "blue"
        },
        {
            "id": 5,
            "image_path": os.path.join(images_dir, "product_5.jpg"),
            "review_text": "Waste of money. Very disappointed.",
            "sentiment": "negative",
            "image_color": "black"
        }
    ]

    # Generate images
    for item in data:
        img = Image.new('RGB', (224, 224), color=item["image_color"])
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Product {item['id']}", fill="white")
        img.save(item["image_path"])
        print(f"Created {item['image_path']}")

    # Save CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(dataset_dir, "reviews.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created dataset at {csv_path}")

if __name__ == "__main__":
    create_mock_dataset()
