import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title="Waste Classifier", layout="wide")
st.title("🗑️ Waste Image Classification")
st.write("Upload a waste image to classify it")

# Define class names
CLASS_NAMES = [
    "Cardboard", "Food Organics", "Glass", "Metal",
    "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"
]

# Function to load trained model
@st.cache_resource
def load_model():
    """Load trained model from saved_models/best_model.pth"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace last layer with our classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 9)
    )
    
    # Load saved weights
    model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

# Load the model
model, device = load_model()
# Function to make predictions
def predict(image, model, device):
    """Make prediction on image"""
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Convert image to tensor
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return probabilities
# File uploader
uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display image
    st.image(image, caption="Your uploaded image", use_column_width=True)
    
    # Make prediction
    probs = predict(image, model, device)
    
    # Get results
    predicted_class_idx = probs.argmax(dim=1).item()
    confidence = probs[0, predicted_class_idx].item() * 100
    
    # Display prediction
    st.write(f"**Predicted Class:** {CLASS_NAMES[predicted_class_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")