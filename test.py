import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from fairface import FairFaceDataset
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load validation dataset
val_dataset = FairFaceDataset(root_dir="Z:\\data\\fairface-img-margin025-trainval", train=False, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define model
class MultiTaskResNet(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2, num_race_classes=7):
        super(MultiTaskResNet, self).__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        feature_dim = 512

        self.age_head = nn.Linear(feature_dim, num_age_classes)
        self.gender_head = nn.Linear(feature_dim, num_gender_classes)
        self.race_head = nn.Linear(feature_dim, num_race_classes)

    def forward(self, x):
        x = self.backbone(x)
        age_out = self.age_head(x)
        gender_out = self.gender_head(x)
        race_out = self.race_head(x)
        return age_out, gender_out, race_out

# Load model
model = MultiTaskResNet().to(device)
#model = ModuleValidator.fix(model)
#ModuleValidator.validate(model, strict=False)

state_dict = torch.load("resnet_fairface.pth", map_location=device)
#remove_prefix = '_module.'
#state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Evaluation function
def evaluate_model():
    correct_age, correct_gender, correct_race = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for sample in tqdm(val_loader):
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
    evaluate_model()
