import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

DATA_DIR = "./data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

class_names = dataset.classes

total = len(dataset)

train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size]
)

test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)


model = models.resnet50(
    weights=None
)

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 9)
)

model.load_state_dict(
    torch.load(
        "saved_models/best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


acc = accuracy_score(
    all_labels,
    all_preds
)

print(f"\nTest Accuracy: {acc:.4f}")


print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )
)


cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "saved_models/confusion_matrix.png"
)

plt.show()