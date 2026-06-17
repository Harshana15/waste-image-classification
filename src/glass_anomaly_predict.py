import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# IMPORTANT:
# Training showed:
# broken = 0
# normal = 1

class_names = [
    "Broken",
    "Normal"
]

model = models.resnet50(weights=None)

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

model.load_state_dict(
    torch.load(
        "saved_models/glass_anomaly_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

IMAGE_PATH = input("Enter image path: ")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():

    outputs = model(tensor)

    probs = torch.softmax(
        outputs,
        dim=1
    )

    pred = outputs.argmax(1).item()

print(
    f"\nPrediction: {class_names[pred]}"
)

print(
    f"Confidence: {probs[0][pred].item()*100:.2f}%"
)