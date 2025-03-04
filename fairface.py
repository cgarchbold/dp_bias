import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FairFaceDataset(Dataset):
    """FairFace dataset."""

    def __init__(self, root_dir, train=True, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
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