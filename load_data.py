import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fairface import FairFaceDataset 

def get_dataloaders(batch_size):
    # Define transformations
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset, for now location is hardcoded
    train_dataset = FairFaceDataset(root_dir="Z:\\data\\fairface-img-margin025-trainval", train=True, transform=transform)
    val_dataset = FairFaceDataset(root_dir="Z:\\data\\fairface-img-margin025-trainval", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
        