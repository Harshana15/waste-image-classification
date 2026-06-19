import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

st.set_page_config(
    page_title="Waste Classification Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🗑️ Waste Image Classification Pro")
st.write("Complete ML Solution with Anomaly Detection & XAI")

WASTE_CLASSES = [
    "Cardboard", "Food Organics", "Glass", "Metal",
    "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"
]

GLASS_ANOMALY_CLASSES = ["Normal Glass", "Anomalous Glass"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

def predict_waste(image_pil):
    """Predict waste category"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = main_model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs

def predict_glass_anomaly(image_pil):
    """Predict glass anomaly"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = glass_anomaly_model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs

def predict_for_lime(images):
    """Prediction function for LIME"""
    batch = []
    for img in images:
        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor = transform(pil_img)
        batch.append(tensor)
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        outputs = main_model(batch)
        probs = torch.softmax(outputs, dim=1)
    return probs.cpu().numpy()

def generate_gradcam(image_pil, predicted_class_idx):
    """Generate Grad-CAM"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    target_layer = main_model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    output = main_model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    main_model.zero_grad()
    output[:, pred_class].backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations.squeeze(0)

    for i in range(len(pooled_gradients)):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    img_array = np.array(image_pil.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_color * 0.4 + img_array

    # Clear hooks
    for hook in target_layer._forward_hooks.values():
        hook.remove()
    for hook in target_layer._backward_hooks.values():
        hook.remove()

    return superimposed.astype(np.uint8)

def generate_lime(image_pil, predicted_class_idx):
    """Generate LIME explanation"""
    image_pil = image_pil.convert("RGB")
    image_np = np.array(image_pil)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        predict_for_lime,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    return mark_boundaries(temp / 255.0, mask)

# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_main_model():
    """Load main waste classification model"""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 9)
    )
    model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_glass_anomaly_model():
    """Load glass anomaly detection model"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    try:
        model.load_state_dict(torch.load("saved_models/glass_anomaly_model.pth", map_location=device))
    except:
        st.warning("Glass anomaly model not found!")
    model.to(device)
    model.eval()
    return model

main_model = load_main_model()
glass_anomaly_model = load_glass_anomaly_model()

st.success("✅ All models loaded successfully!")

# ==========================================
# UNIFIED DASHBOARD
# ==========================================

st.subheader("📤 Upload Image to Analyze")

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image_pil)

    # ==========================================
    # 1. WASTE CLASSIFICATION
    # ==========================================

    st.divider()
    st.header("🗑️ Waste Classification")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with col2:
        probs = predict_waste(image_pil)
        predicted_class_idx = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class_idx].item() * 100

        st.metric(
            "Predicted Class",
            WASTE_CLASSES[predicted_class_idx],
            f"{confidence:.2f}% confidence"
        )

        st.write("**Class Probabilities:**")
        prob_data = {
            WASTE_CLASSES[i]: probs[0, i].item() * 100
            for i in range(len(WASTE_CLASSES))
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(list(prob_data.keys()), list(prob_data.values()))

        max_idx = list(prob_data.values()).index(max(prob_data.values()))
        bars[max_idx].set_color('#1f77b4')
        for i, bar in enumerate(bars):
            if i != max_idx:
                bar.set_color('#d3d3d3')

        ax.set_xlabel('Probability (%)')
        ax.set_xlim(0, 100)

        for i, v in enumerate(prob_data.values()):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')

        st.pyplot(fig, use_container_width=True)

    # ==========================================
    # 2. ANOMALY DETECTION (If Glass)
    # ==========================================

    if predicted_class_idx == 2:  # Glass
        st.divider()
        st.header("🔍 Anomaly Detection (Glass)")

        anomaly_probs = predict_glass_anomaly(image_pil)
        anomaly_idx = anomaly_probs.argmax(dim=1).item()
        anomaly_confidence = anomaly_probs[0, anomaly_idx].item() * 100

        col1, col2 = st.columns(2)

        with col1:
            if anomaly_idx == 0:
                st.success(f"✅ {GLASS_ANOMALY_CLASSES[0]}")
            else:
                st.error(f"⚠️ {GLASS_ANOMALY_CLASSES[1]}")
            st.metric("Confidence", f"{anomaly_confidence:.2f}%")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                GLASS_ANOMALY_CLASSES,
                [anomaly_probs[0, 0].item() * 100, anomaly_probs[0, 1].item() * 100]
            )
            bars[anomaly_idx].set_color('#28a745' if anomaly_idx == 0 else '#dc3545')
            bars[1 - anomaly_idx].set_color('#d3d3d3')
            ax.set_xlabel('Probability (%)')
            ax.set_xlim(0, 100)
            for i, v in enumerate([anomaly_probs[0, 0].item() * 100, anomaly_probs[0, 1].item() * 100]):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            st.pyplot(fig, use_container_width=True)

    # ==========================================
    # 3. EXPLAINABLE AI (Grad-CAM + LIME)
    # ==========================================

    st.divider()
    st.header("✨ Explainable AI (XAI)")
    st.write("Understanding model predictions with Grad-CAM and LIME")

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.subheader("Grad-CAM Visualization")
        st.write("Red = High importance | Blue = Low importance")

        with st.spinner("Generating Grad-CAM..."):
            try:
                main_model.eval()
                gradcam_img = generate_gradcam(image_pil, predicted_class_idx)
                st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM Error: {str(e)}")

    with xai_col2:
        st.subheader("LIME Explanation")
        st.write("Highlights important regions for prediction")

        with st.spinner("Generating LIME..."):
            try:
                lime_img = generate_lime(image_pil, predicted_class_idx)
                st.image(lime_img, caption="LIME Explanation", use_column_width=True)
            except Exception as e:
                st.error(f"LIME Error: {str(e)}")

    st.success(
        f"✅ Complete! Prediction: **{WASTE_CLASSES[predicted_class_idx]}** ({confidence:.2f}%)"
    )

else:
    st.info("👆 Upload an image to analyze")
