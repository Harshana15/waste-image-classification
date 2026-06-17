import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


full_dataset = datasets.ImageFolder(
    root="./data",
    transform=train_transform
)

print("All Classes:", full_dataset.classes)

glass_idx = full_dataset.class_to_idx["Glass"]
plastic_idx = full_dataset.class_to_idx["Plastic"]


class GlassPlasticDataset(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset

        self.samples = []

        for path, label in dataset.samples:

            if label == glass_idx:
                self.samples.append((path, 0))

            elif label == plastic_idx:
                self.samples.append((path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = self.dataset.loader(path)

        image = self.dataset.transform(image)

        return image, label

dataset = GlassPlasticDataset(full_dataset)

print("Glass + Plastic Images:", len(dataset))


total = len(dataset)

train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print(
    f"Train: {len(train_set)} | "
    f"Val: {len(val_set)} | "
    f"Test: {len(test_set)}"
)

train_loader = DataLoader(
    train_set,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_set,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V1
)

for param in model.parameters():
    param.requires_grad = False

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


EPOCHS = 15

best_val_acc = 0.0

os.makedirs(
    "saved_models",
    exist_ok=True
)

for epoch in range(EPOCHS):


    model.train()

    train_correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()

    train_acc = train_correct / len(train_set)

    model.eval()

    val_correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

    val_acc = val_correct / len(val_set)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "saved_models/glass_plastic_model.pth"
        )

        print(
            f"Best Model Saved! "
            f"Val Acc: {val_acc:.4f}"
        )

print("\nTraining Complete!")
print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)