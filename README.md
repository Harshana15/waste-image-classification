# waste-image-classification

# Project Goal
The goal of this project is to develop a comprehensive machine learning solution for waste sorting. It includes waste classification, anomaly detection for glass and plastic, and explainable AI techniques (Grad-CAM and LIME) to validate model predictions.

# Features
Classification of waste into 9 categories: \n \n
→ Cardboard \n \n
→ Food Organics \n \n
→ Glass \n \n
→ Metal \n \n
→ Miscellaneous Trash \n \n
→ Paper \n \n
→ Plastic \n \n
→ Textile Trash \n \n
→ Vegetation \n \n

# Plastic vs Glass
→ Identifies if the classification is Plastic or Glass:\n \n
→ Glass and Plastic are often confused, so this step ensures accurate classification.\n \n

# Anomaly Detection
Detects defective waste: \n \n
→ Glass: Normal vs Broken \n \n
→ Identifies quality issues \n \n

# Explainability
Understand model decisions: \n \n
→ Grad-CAM heatmaps\n \n
→ LIME explanations\n \n

# Technical Stack
Model: ResNet50 (Transfer Learning)\n \n
Framework: PyTorch \n \n
Frontend: Streamlit \n \n
XAI: Grad-CAM + LIME\n \n
Accuracy: 80% on waste classification\n \n

# Model Performance
Classification Accuracy : 80% \n \n
Categories : 9 \n \n
Glass vs Plastic : 1 \n \n
Anomaly Models (Broken vs Normal Glass) : 1\n \n
