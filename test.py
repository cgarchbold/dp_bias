import os
import csv
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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

    model_pth = os.path.join('./runs/',args.exp_name,'model.pth')

    state_dict = torch.load(model_pth, map_location=device)
    if args.private:
        remove_prefix = '_module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluation function
    correct_age, correct_gender, correct_race = 0, 0, 0
    total_samples = 0

    age_preds_list, age_labels_list = [], []
    gender_preds_list, gender_labels_list = [], []
    race_preds_list, race_labels_list = [], []

    with torch.no_grad():
        for sample in tqdm(test_loader):
            images = sample['image']
            age_labels, gender_labels, race_labels = sample['labels']
            images, age_labels, gender_labels, race_labels = images.to(device), age_labels.to(device), gender_labels.to(device), race_labels.to(device)
            
            age_preds, gender_preds, race_preds = model(images)
            
            age_preds_list.extend(torch.argmax(age_preds, dim=1).cpu().numpy())
            age_labels_list.extend(age_labels.cpu().numpy())
            
            gender_preds_list.extend(torch.argmax(gender_preds, dim=1).cpu().numpy())
            gender_labels_list.extend(gender_labels.cpu().numpy())
            
            race_preds_list.extend(torch.argmax(race_preds, dim=1).cpu().numpy())
            race_labels_list.extend(race_labels.cpu().numpy())

    # Compute metrics
    metrics = {
        "Age": {
            "Accuracy": accuracy_score(age_labels_list, age_preds_list) * 100,
            "F1-score": f1_score(age_labels_list, age_preds_list, average='macro') * 100,
            #"AUC-ROC": roc_auc_score(age_labels_list, age_preds_list, multi_class='ovr') * 100
        },
        "Gender": {
            "Accuracy": accuracy_score(gender_labels_list, gender_preds_list) * 100,
            "F1-score": f1_score(gender_labels_list, gender_preds_list, average='macro') * 100,
            #"AUC-ROC": roc_auc_score(gender_labels_list, gender_preds_list) * 100
        },
        "Race": {
            "Accuracy": accuracy_score(race_labels_list, race_preds_list) * 100,
            "F1-score": f1_score(race_labels_list, race_preds_list, average='macro') * 100,
            #"AUC-ROC": roc_auc_score(race_labels_list, race_preds_list, multi_class='ovr') * 100
        }
    }

    # Compute per-class accuracy for Race and Gender
    race_classes = sorted(set(race_labels_list))
    gender_classes = sorted(set(gender_labels_list))
    
    race_acc = {f"Race {cls}": f1_score([1 if x == cls else 0 for x in race_labels_list],
                                               [1 if x == cls else 0 for x in race_preds_list], average='macro') * 100 for cls in race_classes}
    
    gender_acc = {f"Gender {cls}": f1_score([1 if x == cls else 0 for x in gender_labels_list],
                                                  [1 if x == cls else 0 for x in gender_preds_list], average='macro') * 100 for cls in gender_classes}
    
    results_path = os.path.join('./runs/', args.exp_name, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("Evaluation Results:\n")
        for task, results in metrics.items():
            f.write(f"{task}:\n")
            for metric, value in results.items():
                f.write(f"  {metric}: {value:.2f}%\n") 
        
        f.write("\nRace Breakdown:\n")
        for cls, acc in race_acc.items():
            f.write(f"  {cls}: {acc:.2f}%\n")
        
        f.write("\nGender Breakdown:\n")
        for cls, acc in gender_acc.items():
            f.write(f"  {cls}: {acc:.2f}%\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiTaskResNet with optional differential privacy.")
    parser.add_argument("--exp_name", type=str, required=True, help="Path to save experiment")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for testing")
    parser.add_argument("--private", action='store_true', help="Enable differential private model")
    parser.add_argument("--pretrained", action='store_true', help="Enable imagenet weights")
    parser.add_argument("--freeze_backbone", action='store_true', help="Enable freezeing backbone weights")

    args = parser.parse_args()
    
    test(args=args)
