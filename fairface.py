import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FairFaceDataset(Dataset):
    """FairFace dataset."""

    def __init__(self, root_dir, train=True, transform=None, biased=False, seed=42):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        np.random.seed(seed) 

        if train:
            csv_file = os.path.join(root_dir, 'fairface_label_train.csv')
            self.anno = pd.read_csv(csv_file)
        else:
            csv_file = os.path.join(root_dir, 'fairface_label_val.csv')
            self.anno = pd.read_csv(csv_file)

        self.enc_age = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4, "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8}
        self.enc_gender = {"Male": 0, "Female": 1}
        self.enc_race = {"White": 0, "Latino_Hispanic": 1, "Indian": 2, "East Asian": 3, "Black": 4, "Southeast Asian": 5, "Middle Eastern": 6}

        # Convert categorical labels to numerical labels
        self.anno["age"] = self.anno["age"].map(self.enc_age)
        self.anno["gender"] = self.anno["gender"].map(self.enc_gender)
        self.anno["race"] = self.anno["race"].map(self.enc_race)

        # Apply bias if needed
        if biased:
            self.apply_bias()

    def apply_bias(self):
        """Randomly removes half of all women and half of each specified race (Black, Indian, Middle Eastern)."""
        
        # Remove half of women
        women_indices = self.anno[self.anno["gender"] == 1].index
        women_to_remove = np.random.choice(women_indices, size=len(women_indices) // 2, replace=False)

        # Remove half of each specified race individually
        races_to_remove = [2, 4, 6]  # Indian, Black, Middle Eastern
        race_removals = []
        
        for race in races_to_remove:
            race_indices = self.anno[self.anno["race"] == race].index
            race_to_remove = np.random.choice(race_indices, size=len(race_indices) // 2, replace=False)
            race_removals.extend(race_to_remove)

        # Combine indices to remove
        to_remove = np.concatenate([women_to_remove, race_removals])
        
        # Drop selected rows
        self.anno = self.anno.drop(to_remove).reset_index(drop=True)

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.anno.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        labels = self.anno.iloc[idx, 1:4].to_list()
        sample = {'image': image, 'labels': labels}

        return sample
    
    def decode_labels(self, age, gender, race):
        """
        Converts numerical labels back to categorical labels.
        
        Arguments:
            age (int): Encoded age label.
            gender (int): Encoded gender label.
            race (int): Encoded race label.
        
        Returns:
            tuple: (age_str, gender_str, race_str)
        """
        return self.dec_age.get(age, "Unknown"), self.dec_gender.get(gender, "Unknown"), self.dec_race.get(race, "Unknown")