import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
from transformers import CLIPImageProcessor

class GroceryDataset(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.csv_path = "/scratch/ys6310/Mario/dataset/Grocery/Grocery.csv"
        self.images_folder = "/scratch/ys6310/Mario/dataset/Grocery/GroceryImages"
        self.data = pd.read_csv(self.csv_path, usecols=['id', 'text', 'second_category', 'label'])
        self.data = self.data[self.data['id'].apply(self._image_exists)].reset_index(drop=True)
        all_categories = sorted(self.data['second_category'].unique()) 
        self.prompt = "\nQuestion: Given the image and the text description of an item from the 'Grocery & Gourmet Food' category on Amazon, determine which of the following subcategories the item belongs to: "
        self.prompt += ", ".join(all_categories)
        self.image_processor = CLIPImageProcessor.from_pretrained("/scratch/ys6310/llava-1.5-13b-hf")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError("Slicing is not supported for this dataset.")

        # Retrieve the row corresponding to the index
        row = self.data.iloc[idx]

        # Extract text and labels
        text = row['text']
        label = row['label']
        category = row['second_category']

        # Load the corresponding image
        image_id = row['id']
        image_path = os.path.join(self.images_folder, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image = self.image_processor(images=image, return_tensors="pt")

        return {
            'text': text,
            'category': category,
            'label': label,
            'image': image,
            'prompt': self.prompt
        }

    def _image_exists(self, image_id):
        """
        Check if the image exists for a given image ID.
        """
        image_path = os.path.join(self.images_folder, f"{image_id}.jpg")
        return os.path.exists(image_path)
    
    def dataset_stats(self):
        """
        Returns the number of categories and total number of data points.
        """
        num_categories = self.data['label'].nunique()
        total_data = len(self.data)
        return num_categories, total_data
        
    def get_idx_split(self, train_ratio=0.1, val_ratio=0.1, seed=0):
        """
        Randomly splits the dataset indices into train, validation, and test sets.

        Args:
            train_ratio (float): The proportion of training data.
            val_ratio (float): The proportion of validation data.
            seed (int, optional): Random seed for reproducibility.
        
        Returns:
            dict: A dictionary with keys 'train', 'val', 'test' containing respective indices.
        """
        nodes_num = len(self.data)
        indices = np.arange(nodes_num)
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Shuffle the indices
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        # Create splits
        train_ids = indices[:train_size]
        val_ids = indices[train_size:train_size + val_size]
        test_ids = indices[train_size + val_size:]

        # train_ids = indices[:100]
        # val_ids = indices[100:150]
        # test_ids = indices[150:200]

        return {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }

# Example usage
# if __name__ == "__main__":
#     dataset = GroceryDataset()
#     sample = dataset[0]  # Access the first sample

#     print("Text:", sample['text'])
#     print("Category:", sample['category'])
#     print("Label:", sample['label'])
#     num_categories, total_data = dataset.dataset_stats()
#     print(f"Number of categories: {num_categories}")
#     print(f"Total data points: {total_data}")
