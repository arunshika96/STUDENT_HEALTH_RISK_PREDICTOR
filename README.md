# Student Health Risk Predictor

A Flask-based web application that predicts student health risk categories (**at-risk, fit, or unhealthy**) based on lifestyle and biometric indicators. This project was developed as part of the **CIS 6005 Computational Intelligence** assignment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--learn-orange)

---

## Table of Contents
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [How to Run the Web App](#how-to-run-the-web-app)
- [How to Train the Models](#how-to-train-the-models)
- [Features](#features)

---

## Project Structure

STUDENT_HEALTH_RISK_PREDICTOR/
├── health_risk_app/ # The Flask web application
│ ├── app.py # Main Flask backend
│ ├── models/ # Contains pre-trained .pkl model files
│ │ ├── best_model.pkl # Best XGBoost model used by the app
│ │ ├── xgboost.pkl
│ │ └── ...
│ ├── static/ # CSS and static assets
│ ├── templates/ # HTML templates (index.html)
│ ├── .env # Local environment variables (ignored by Git)
│ ├── .gitignore # Git ignore rules for the web app
│ └── requirements.txt # Web app dependencies
├── outputs/ # Training logs and visualizations
├── submission.csv # Generated assignment output data
├── train_model.py # Main script to train and save all models
├── model_training.py # Core ML training logic
├── predict.py # Script to run predictions via CLI
├── compress_model.py # Utility to compress model sizes
├── requirements.txt # Root-level dependencies (ML + Web)
├── .gitattributes # Git LFS config for large .pkl files
└── README.md # Project documentation


---

## Technologies Used
- **Web Framework:** Python (Flask)
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Version Control:** Git LFS (for managing large model files)

---

## Installation & Setup

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/arunshika96/STUDENT_HEALTH_RISK_PREDICTOR.git
   
