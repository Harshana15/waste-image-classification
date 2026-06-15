# pip install opencv-python pillow matplotlib

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load Model
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class_names = [
    'Cardboard',
    'Food Organics',
    'Glass',
    'Metal',
    'Miscellaneous Trash',
    'Paper',
    'Plastic',
    'Textile Trash',
    'Vegetation'
]

model = models.resnet50(weights=None)

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

# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# --------------------------------------------------
# Select Image
# --------------------------------------------------


''' import glob
import random

glass_images = glob.glob("data/Glass/*.jpg")

IMAGE_PATH = random.choice(glass_images)

print("Selected:", IMAGE_PATH) ''' 

IMAGE_PATH = input("Enter image path: ")

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# --------------------------------------------------
# Hooks
# --------------------------------------------------

gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer = model.layer4[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# --------------------------------------------------
# Forward Pass
# --------------------------------------------------

output = model(input_tensor)

pred_class = output.argmax(dim=1).item()

print(
    "Predicted:",
    class_names[pred_class]
)

# --------------------------------------------------
# Backward Pass
# --------------------------------------------------

model.zero_grad()

output[:, pred_class].backward()

# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------

pooled_gradients = torch.mean(
    gradients,
    dim=[0,2,3]
)

activations = activations.squeeze(0)

for i in range(len(pooled_gradients)):
    activations[i,:,:] *= pooled_gradients[i]

heatmap = torch.mean(
    activations,
    dim=0
).cpu().detach().numpy()

heatmap = np.maximum(heatmap, 0)

heatmap /= heatmap.max()

# --------------------------------------------------
# Overlay Heatmap
# --------------------------------------------------

img = cv2.imread(IMAGE_PATH)

img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

heatmap = cv2.resize(
    heatmap,
    (img.shape[1], img.shape[0])
)

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)

superimposed = heatmap * 0.4 + img

# --------------------------------------------------
# Display
# --------------------------------------------------

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(superimposed.astype(np.uint8))
plt.title(
    f"Grad-CAM\nPrediction: {class_names[pred_class]}"
)
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "saved_models/gradcam_result.png"
)

plt.show()