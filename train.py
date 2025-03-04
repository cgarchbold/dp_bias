import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from model import MultiTaskResNet
from load_data import get_dataloaders
import argparse
import os

import warnings
warnings.filterwarnings('ignore')


def train(epochs, lr, batch_size, exp_name, private = False, epsilon=10, delta=0.0001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = MultiTaskResNet().to(device)
    if private:
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if private:
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon =epsilon,
            target_delta = delta, # 1/ total_datapoints
            epochs = epochs,
            max_grad_norm=1.0
        )


    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for sample in tqdm(train_loader):
            images = sample['image']
            age_labels, gender_labels, race_labels = sample['labels']
            images, age_labels, gender_labels, race_labels = images.to(device), age_labels.to(device), gender_labels.to(device), race_labels.to(device)

            optimizer.zero_grad()
            age_preds, gender_preds, race_preds = model(images)

            loss_age = criterion(age_preds, age_labels)
            loss_gender = criterion(gender_preds, gender_labels)
            loss_race = criterion(race_preds, race_labels)

            loss = loss_age + loss_gender + loss_race  # Multi-task loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join('./runs/',exp_name,'model.pth'))
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

    args = parser.parse_args()

    # Create directory for experiment
    os.makedirs(os.path.join("runs", args.exp_name), exist_ok=True)
    
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, exp_name=args.exp_name, private=args.private, epsilon=args.epsilon, delta=args.delta)