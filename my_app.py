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
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="WasteVisionX: Explainable Classification and Anomaly Detection for Waste Sorting",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Top Header with About Link
col1, col2 = st.columns([0.85, 0.15])

with col1:
    st.title("Waste Image Classification")
  #  st.write("Complete ML Solution with Anomaly Detection & XAI")

with col2:
    st.write("")
    st.write("")
    if st.button("About", use_container_width=True, key="about_btn"):
        st.session_state.show_about = True

WASTE_CLASSES = [
    "Cardboard", "Food Organics", "Glass", "Metal",
    "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"
]

# IMPORTANT: Folders are alphabetically ordered: broken=0, normal=1
GLASS_ANOMALY_CLASSES = ["Anomalous Glass (Broken)", "Normal Glass"]
PLASTIC_ANOMALY_CLASSES = ["Anomalous Plastic", "Normal Plastic"]

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

def predict_plastic_anomaly(image_pil):
    """Predict plastic anomaly"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = plastic_anomaly_model(image_tensor)
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
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    output = main_model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    main_model.zero_grad()
    output[:, pred_class].backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations.squeeze(0)

    # Use broadcasting instead of loop
    activations = activations * pooled_gradients.view(-1, 1, 1)

    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    img_array = np.array(image_pil.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_color * 0.4 + img_array

    # Remove hooks properly
    fwd_handle.remove()
    bwd_handle.remove()

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
        num_samples=25
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    return mark_boundaries(temp / 255.0, mask)

def predict_for_lime_glass_anomaly(images):
    """Prediction function for LIME - Glass Anomaly"""
    batch = []
    for img in images:
        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor = transform(pil_img)
        batch.append(tensor)
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        outputs = glass_anomaly_model(batch)
        probs = torch.softmax(outputs, dim=1)
    return probs.cpu().numpy()

def predict_for_lime_plastic_anomaly(images):
    """Prediction function for LIME - Plastic Anomaly"""
    batch = []
    for img in images:
        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor = transform(pil_img)
        batch.append(tensor)
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        outputs = plastic_anomaly_model(batch)
        probs = torch.softmax(outputs, dim=1)
    return probs.cpu().numpy()

def generate_gradcam_anomaly(image_pil, model_to_use, model_name):
    """Generate Grad-CAM for anomaly models"""
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    target_layer = model_to_use.layer4[-1]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    with torch.enable_grad():
        output = model_to_use(image_tensor)
        pred_class = output.argmax(dim=1).item()

        model_to_use.zero_grad()
        output[:, pred_class].backward()

    if gradients is None or activations is None:
        fwd_handle.remove()
        bwd_handle.remove()
        raise RuntimeError("Failed to capture gradients for Grad-CAM")

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations.squeeze(0)
    activations = activations * pooled_gradients.view(-1, 1, 1)

    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    img_array = np.array(image_pil.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_color * 0.4 + img_array

    fwd_handle.remove()
    bwd_handle.remove()

    return superimposed.astype(np.uint8)

def generate_lime_anomaly(image_pil, predict_func):
    """Generate LIME explanation for anomaly models"""
    image_pil = image_pil.convert("RGB")
    image_np = np.array(image_pil)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        predict_func,
        top_labels=1,
        hide_color=0,
        num_samples=25
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

@st.cache_resource
def load_plastic_anomaly_model():
    """Load plastic anomaly detection model"""
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
        model.load_state_dict(torch.load("saved_models/glass_plastic_model.pth", map_location=device))
    except:
        st.warning("Plastic anomaly model not found!")
    model.to(device)
    model.eval()
    return model

main_model = load_main_model()
glass_anomaly_model = load_glass_anomaly_model()
plastic_anomaly_model = load_plastic_anomaly_model()

#st.success("All models loaded successfully!")

# ==========================================
# UNIFIED DASHBOARD
# ==========================================

st.subheader("Upload Image to Analyze")

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
    st.header("Waste Classification")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    with col2:
        probs = predict_waste(image_pil)
        predicted_class_idx = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class_idx].item() * 100

    # Track final prediction (may be updated by verification)
    final_predicted_class_idx = predicted_class_idx
    final_confidence = confidence

    with col2:
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
    # 2. ANOMALY DETECTION (If Glass or Plastic)
    # ==========================================

    if predicted_class_idx == 2:  # Glass
        st.divider()
        st.header("Verification: Glass vs Plastic")
        st.write("First verifying this is Glass, not Plastic")

        # Use glass_plastic_model (0=Glass, 1=Plastic)
        gp_probs = predict_plastic_anomaly(image_pil)
        gp_idx = gp_probs.argmax(dim=1).item()
        gp_confidence = gp_probs[0, gp_idx].item() * 100

        col1, col2 = st.columns(2)

        with col1:
            if gp_idx == 0:  # Glass confirmed
                st.success(f"Confirmed: This is Glass")
            else:  # Actually Plastic
                st.warning(f"This might be Plastic (Not Glass)")
            st.metric("Confidence", f"{gp_confidence:.2f}%")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                ["Glass", "Plastic"],
                [gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]
            )
            bars[gp_idx].set_color('#28a745' if gp_idx == 0 else '#dc3545')
            bars[1 - gp_idx].set_color('#d3d3d3')
            ax.set_xlabel('Probability (%)')
            ax.set_xlim(0, 100)
            for i, v in enumerate([gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            st.pyplot(fig, use_container_width=True)


    elif predicted_class_idx == 6:  # Plastic
        st.divider()
        st.header("Verification: Plastic vs Glass")
        st.write("Confirming this is Plastic, not Glass")

        # Use glass_plastic_model (0=Glass, 1=Plastic)
        gp_probs = predict_plastic_anomaly(image_pil)
        gp_idx = gp_probs.argmax(dim=1).item()
        gp_confidence = gp_probs[0, gp_idx].item() * 100

        # Update final prediction based on verification
        if gp_idx == 0:  # Verification says Glass
            final_predicted_class_idx = 2  # Glass index
            final_confidence = gp_confidence
        # else: keep as Plastic (gp_idx == 1)

        col1, col2 = st.columns(2)

        with col1:
            if gp_idx == 1:  # Plastic
                st.success(f"This is Plastic (Not Glass)")
            else:  # Glass
                st.warning(f"This might be Glass")
            st.metric("Confidence", f"{gp_confidence:.2f}%")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                ["Glass", "Plastic"],
                [gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]
            )
            bars[gp_idx].set_color('#28a745')
            bars[1 - gp_idx].set_color('#d3d3d3')
            ax.set_xlabel('Probability (%)')
            ax.set_xlim(0, 100)
            for i, v in enumerate([gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            st.pyplot(fig, use_container_width=True)

    else:
        st.divider()
        st.header("Anomaly Detection")
        st.info("Anomaly detection is only available for Glass and Plastic categories.")

    # ==========================================
    # 2B. ANOMALY DETECTION - ALWAYS DISPLAY
    # ==========================================

    st.divider()
    st.header("Anomaly Detection Results")

    if final_predicted_class_idx == 2:  # Glass
        st.write("Detecting if glass is broken or normal")
        anomaly_probs = predict_glass_anomaly(image_pil)
        anomaly_idx = anomaly_probs.argmax(dim=1).item()
        anomaly_confidence = anomaly_probs[0, anomaly_idx].item() * 100

        col1, col2 = st.columns(2)

        with col1:
            if anomaly_idx == 1:  # Normal Glass
                st.success(f"Status: {GLASS_ANOMALY_CLASSES[1]}")
            else:  # Broken/Anomalous
                st.error(f"Status: {GLASS_ANOMALY_CLASSES[0]}")
            st.metric("Anomaly Confidence", f"{anomaly_confidence:.2f}%")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                GLASS_ANOMALY_CLASSES,
                [anomaly_probs[0, 0].item() * 100, anomaly_probs[0, 1].item() * 100]
            )
            bars[anomaly_idx].set_color('#28a745' if anomaly_idx == 1 else '#dc3545')
            bars[1 - anomaly_idx].set_color('#d3d3d3')
            ax.set_xlabel('Probability (%)')
            ax.set_xlim(0, 100)
            for i, v in enumerate([anomaly_probs[0, 0].item() * 100, anomaly_probs[0, 1].item() * 100]):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            st.pyplot(fig, use_container_width=True)

    elif final_predicted_class_idx == 6:  # Plastic
        st.write("Verifying Plastic vs Glass classification")
        gp_probs = predict_plastic_anomaly(image_pil)
        gp_idx = gp_probs.argmax(dim=1).item()
        gp_confidence = gp_probs[0, gp_idx].item() * 100

        col1, col2 = st.columns(2)

        with col1:
            if gp_idx == 1:  # Plastic confirmed
                st.success(f"Verified: This is Plastic")
            else:  # Actually Glass
                st.warning(f"Alert: This might be Glass")
            st.metric("Verification Confidence", f"{gp_confidence:.2f}%")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                ["Glass", "Plastic"],
                [gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]
            )
            bars[gp_idx].set_color('#28a745')
            bars[1 - gp_idx].set_color('#d3d3d3')
            ax.set_xlabel('Probability (%)')
            ax.set_xlim(0, 100)
            for i, v in enumerate([gp_probs[0, 0].item() * 100, gp_probs[0, 1].item() * 100]):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("Anomaly detection is only available for Glass and Plastic categories.")

    # ==========================================
    # 3. EXPLAINABLE AI (Grad-CAM + LIME) - VALIDATES PREDICTION
    # ==========================================

    st.divider()
    st.header("Explainable AI - XAI Validation")

    # Determine which model to explain (based on final prediction after verification)
    if final_predicted_class_idx == 2:  # Glass
        st.write("Validating Glass Classification & Anomaly Detection with Grad-CAM and LIME")
        explain_model = glass_anomaly_model
        explain_func = predict_for_lime_glass_anomaly
        explain_type = "Glass Anomaly"
        explain_class_idx = 1  # Normal glass for LIME
    elif final_predicted_class_idx == 6:  # Plastic
        st.write("Validating Glass vs Plastic Classification with Grad-CAM and LIME")
        explain_model = plastic_anomaly_model
        explain_func = predict_for_lime_plastic_anomaly
        explain_type = "Plastic vs Glass"
        explain_class_idx = 1  # Plastic for LIME
    else:
        st.write("Analyzing Main Classification with Grad-CAM and LIME")
        explain_model = main_model
        explain_func = predict_for_lime
        explain_type = "Main Classification"
        explain_class_idx = final_predicted_class_idx

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.subheader("Grad-CAM Visualization")
        st.write("Shows which image regions the model focused on (Red = High importance, Blue = Low importance)")

        with st.spinner("Generating Grad-CAM..."):
            try:
                explain_model.eval()
                if explain_type in ["Glass Anomaly", "Plastic vs Glass"]:
                    gradcam_img = generate_gradcam_anomaly(image_pil, explain_model, explain_type)
                else:
                    gradcam_img = generate_gradcam(image_pil, predicted_class_idx)
                st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM Error: {str(e)}")

    with xai_col2:
        st.subheader("LIME Explanation & Validation Confidence")
        st.write("Shows important feature regions. Confidence % comes from LIME prediction for validation.")

        with st.spinner("Generating LIME... (this may take a moment)"):
            try:
                if explain_type in ["Glass Anomaly", "Plastic vs Glass"]:
                    lime_img = generate_lime_anomaly(image_pil, explain_func)
                    # Get LIME confidence for validation
                    lime_probs = explain_func(np.expand_dims(np.array(image_pil.resize((224, 224))), 0))
                    lime_confidence = lime_probs[0, explain_class_idx] * 100 if explain_type in ["Glass Anomaly", "Plastic vs Glass"] else final_confidence
                else:
                    lime_img = generate_lime(image_pil, final_predicted_class_idx)
                    # Get LIME confidence for validation
                    lime_probs = explain_func(np.expand_dims(np.array(image_pil.resize((224, 224))), 0))
                    lime_confidence = lime_probs[0, final_predicted_class_idx] * 100

                # Update final confidence based on XAI validation
                final_confidence = lime_confidence

                st.image(lime_img, caption="LIME Explanation", use_container_width=True)
                st.metric("XAI Validation Confidence", f"{lime_confidence:.2f}%")
            except Exception as e:
                st.error(f"LIME Error: {str(e)}")

    # Final Prediction - Based on XAI Validation
    st.divider()
    st.success(
        f"Final Prediction (XAI Validated): **{WASTE_CLASSES[final_predicted_class_idx]}** with {final_confidence:.2f}% confidence"
    )

else:
    st.info(" Upload an image to analyze")

# ==========================================
# ABOUT SECTION (Sidebar)
# ==========================================

if st.session_state.get("show_about", False):
    st.divider()
    st.header("About This Project")

    st.subheader("Project Goal")
    st.write(
        "Automatically classify waste images into 9 categories using Deep Learning "
        "with explainable AI for transparency and anomaly detection capabilities."
    )

    st.subheader("Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**Classification**\n\n"
            "Classifies waste into 9 categories:\n\n"
            "• Cardboard\n"
            "• Food Organics\n"
            "• Glass\n"
            "• Metal\n"
            "• Miscellaneous Trash\n"
            "• Paper\n"
            "• Plastic\n"
            "• Textile Trash\n"
            "• Vegetation"
        )

    with col2:
        st.info(
            "**Anomaly Detection**\n\n"
            "Detects defective waste:\n\n"
            "• Glass: Normal vs Broken\n"
            "• Plastic: Normal vs Damaged\n\n"
            "Identifies quality issues"
        )

    with col3:
        st.info(
            "**Explainability**\n\n"
            "Understand model decisions:\n\n"
            "• Grad-CAM heatmaps\n"
            "• LIME explanations\n\n"
            "Transparent AI!"
        )

    st.subheader(" Technical Stack")
    st.write(
        "- **Model:** ResNet50 (Transfer Learning)\n"
        "- **Framework:** PyTorch\n"
        "- **Frontend:** Streamlit\n"
        "- **XAI:** Grad-CAM + LIME\n"
        "- **Accuracy:** 77% on waste classification"
    )

    st.subheader(" Team Members")
    team_info = """
    - **Sindhuja** 
    - **Shan** 
    - **Harshana** 

    **Project:** Waste Image Classification with Explainable AI
    """
    st.write(team_info)

    st.subheader("Model Performance")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Classification Accuracy", "77%")

    with metric_col2:
        st.metric("Categories", "9")

    with metric_col3:
        st.metric("Anomaly Models", "2 (Glass + Plastic)")

    if st.button("Close About", use_container_width=True):
        st.session_state.show_about = False
        st.rerun()
