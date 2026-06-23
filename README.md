# waste-image-classification

# Project Goal
The goal of this project is to develop a comprehensive machine learning solution for waste sorting. It includes waste classification, anomaly detection for glass and plastic, and explainable AI techniques (Grad-CAM and LIME) to validate model predictions.

# Features
Classification of waste into 9 categories:
→ Cardboard
→ Food Organics
→ Glass
→ Metal
→ Miscellaneous Trash
→ Paper
→ Plastic
→ Textile Trash
→ Vegetation

# Plastic vs Glass
→ Identifies if the classification is Plastic or Glass:
→ Glass and Plastic are often confused, so this step ensures accurate classification.

# Anomaly Detection
Detects defective waste:
→ Glass: Normal vs Broken
→ Identifies quality issues

# Explainability
Understand model decisions:
→ Grad-CAM heatmaps
→ LIME explanations

# Technical Stack
Model: ResNet50 (Transfer Learning)
Framework: PyTorch
Frontend: Streamlit
XAI: Grad-CAM + LIME
Accuracy: 80% on waste classification

# Model Performance
Classification Accuracy : 80%
Categories : 9
Glass vs Plastic : 1
Anomaly Models (Broken vs Normal Glass) : 1
