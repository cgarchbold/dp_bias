import os
import csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from model import MultiTaskResNet
from load_data import get_dataloaders
from plot import plot_losses

import warnings
warnings.filterwarnings('ignore')

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, apply_bias=args.apply_bias)

    # Initialize model, loss, and optimizer
    model = MultiTaskResNet(pretrained=args.pretrained, freeze_backbone=args.freeze_backbone).to(device)
    if args.private:
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.private:
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon =args.epsilon,
            target_delta = args.delta, # 1/ total_datapoints
            epochs = args.epochs,
            max_grad_norm=1.0
        )


    history = []
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for sample in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            images = sample['image'].to(device)
            age_labels, gender_labels, race_labels = [label.to(device) for label in sample['labels']]
            
            optimizer.zero_grad()
            age_preds, gender_preds, race_preds = model(images)
            
            loss_age = criterion(age_preds, age_labels)
            loss_gender = criterion(gender_preds, gender_labels)
            loss_race = criterion(race_preds, race_labels)
            
            loss = loss_age + loss_gender + loss_race  # Multi-task loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{args.epochs}"):
                images = sample['image'].to(device)
                age_labels, gender_labels, race_labels = [label.to(device) for label in sample['labels']]
                
                age_preds, gender_preds, race_preds = model(images)
                
                loss_age = criterion(age_preds, age_labels)
                loss_gender = criterion(gender_preds, gender_labels)
                loss_race = criterion(race_preds, race_labels)
                
                loss = loss_age + loss_gender + loss_race
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        history.append([epoch+1, train_loss, val_loss])

    # Save loss history to CSV
    csv_path = os.path.join("./runs/", args.exp_name, "loss_history.csv")
    with open(csv_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
        writer.writerows(history)

    plot_losses(history, os.path.join("./runs/", args.exp_name, "loss_plot.png"))

    # Save model
    torch.save(model.state_dict(), os.path.join('./runs/',args.exp_name,'model.pth'))
    print("Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiTaskResNet with optional differential privacy.")
    parser.add_argument("--exp_name", type=str, required=True, help="Path to save experiment")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--private", action='store_true', help="Enable differential privacy")
    parser.add_argument("--epsilon", type=float, default=10, help="Privacy budget (epsilon)")
    parser.add_argument("--delta", type=float, default=0.0001, help="Privacy parameter (delta)")
    parser.add_argument("--pretrained", action='store_true', help="Enable imagenet weights")
    parser.add_argument("--freeze_backbone", action='store_true', help="Enable freezing backbone weights")
    parser.add_argument("--apply_bias", action='store_true', help="Enables biased subset")

    args = parser.parse_args()

    # Create directory for experiment
    os.makedirs(os.path.join("runs", args.exp_name), exist_ok=True)

    # Save the args as a CSV file using pandas
    args_df = pd.DataFrame(vars(args), index=[0])
    save_path = os.path.join("runs", args.exp_name, "experiment_args.csv")
    args_df.to_csv(save_path, index=False)
    
    train(args=args)