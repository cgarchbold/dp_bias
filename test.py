import os
import csv
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from opacus.validators import ModuleValidator
from load_data import get_dataloaders
from model import MultiTaskResNet

import warnings
warnings.filterwarnings('ignore')

def test(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Load model
    model = MultiTaskResNet(pretrained=args.pretrained, freeze_backbone=args.freeze_backbone).to(device)
    if args.private:
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)

    state_dict = torch.load("resnet_fairface.pth", map_location=device)
    #remove_prefix = '_module.'
    #state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluation function
    correct_age, correct_gender, correct_race = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for sample in tqdm(test_loader):
            images = sample['image']
            age_labels, gender_labels, race_labels = sample['labels']
            images, age_labels, gender_labels, race_labels = images.to(device), age_labels.to(device), gender_labels.to(device), race_labels.to(device)
            
            age_preds, gender_preds, race_preds = model(images)
            
            correct_age += (torch.argmax(age_preds, dim=1) == age_labels).sum().item()
            correct_gender += (torch.argmax(gender_preds, dim=1) == gender_labels).sum().item()
            correct_race += (torch.argmax(race_preds, dim=1) == race_labels).sum().item()
            total_samples += images.size(0)

    age_acc = correct_age / total_samples * 100
    gender_acc = correct_gender / total_samples * 100
    race_acc = correct_race / total_samples * 100
    
    print(f"Accuracy - Age: {age_acc:.2f}%, Gender: {gender_acc:.2f}%, Race: {race_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiTaskResNet with optional differential privacy.")
    parser.add_argument("--exp_name", type=str, required=True, help="Path to save experiment")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")

    args = parser.parse_args()
    
    test(args=args)
