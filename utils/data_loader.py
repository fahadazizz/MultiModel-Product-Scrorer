import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ProductReviewDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1]) # image_path is at index 1 in our mock csv (id, image_path, ...)
        # In our mock csv, image_path is absolute or relative? 
        # The mock script saved absolute paths or relative to where script ran?
        # Let's check the mock script. It used os.path.join(images_dir, ...) which is relative to where script ran if dataset_dir was relative.
        # But wait, the mock script used `os.path.join(dataset_dir, "images")`.
        # If I run from root, dataset_dir="dataset".
        # So image_path in CSV is likely "dataset/images/product_1.jpg".
        
        # Let's just use the path from the CSV directly if it exists, else join with root_dir.
        row_img_path = self.data_frame.iloc[idx]['image_path']
        
        if os.path.exists(row_img_path):
             image_path = row_img_path
        else:
             image_path = os.path.join(self.root_dir, row_img_path)

        image = Image.open(image_path).convert('RGB')
        review_text = self.data_frame.iloc[idx]['review_text']
        sentiment = self.data_frame.iloc[idx]['sentiment']
        
        sample = {'image': image, 'review_text': review_text, 'sentiment': sentiment}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def get_dataloader(csv_file, root_dir, batch_size=4, transform=None):
    dataset = ProductReviewDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

import torch
