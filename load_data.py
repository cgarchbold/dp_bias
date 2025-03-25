import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from fairface import FairFaceDataset 

def get_dataloaders(batch_size, apply_bias = False, val_split=0.2, random_seed=42):
    torch.manual_seed(random_seed)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset, for now location is hardcoded
    full_train_dataset = FairFaceDataset(root_dir="Z:\\data\\fairface-img-margin025-trainval", train=True, transform=transform, biased=apply_bias)
    test_dataset = FairFaceDataset(root_dir="Z:\\data\\fairface-img-margin025-trainval", train=False, transform=transform, biased=False)

    # Split into train and validation
    total_size = len(full_train_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
        