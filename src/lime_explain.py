# pip install lime scikit-image

import numpy as np
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

import matplotlib.pyplot as plt

# ----------------------------------------
# Device
# ----------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ----------------------------------------
# Class Names
# ----------------------------------------

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

# ----------------------------------------
# Load Model
# ----------------------------------------

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

# ----------------------------------------
# Image
# ----------------------------------------

IMAGE_PATH = input(
    "Enter image path: "
)

image = Image.open(
    IMAGE_PATH
).convert("RGB")

# ----------------------------------------
# Transform
# ----------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ----------------------------------------
# Prediction Function
# ----------------------------------------

def predict(images):

    batch = []

    for img in images:

        pil_img = Image.fromarray(
            img.astype(np.uint8)
        )

        tensor = transform(
            pil_img
        )

        batch.append(tensor)

    batch = torch.stack(batch)

    batch = batch.to(device)

    with torch.no_grad():

        outputs = model(batch)

        probs = torch.softmax(
            outputs,
            dim=1
        )

    return probs.cpu().numpy()

# ----------------------------------------
# LIME Explainer
# ----------------------------------------

explainer = lime_image.LimeImageExplainer()

image_np = np.array(image)

explanation = explainer.explain_instance(
    image_np,
    predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

# ----------------------------------------
# Get Explanation
# ----------------------------------------

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=False
)

# ----------------------------------------
# Display
# ----------------------------------------

plt.figure(figsize=(8,8))

plt.imshow(
    mark_boundaries(temp/255.0, mask)
)

pred = np.argmax(
    predict(np.expand_dims(image_np,0))
)

plt.title(
    f"LIME Explanation\nPrediction: {class_names[pred]}"
)

plt.axis("off")

plt.savefig(
    "saved_models/lime_result.png"
)

plt.show()