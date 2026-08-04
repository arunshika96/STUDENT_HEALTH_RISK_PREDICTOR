# ==========================================
# Computational Intelligence Assignment
# Model Training
# ==========================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Create folders
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

print("="*60)
print("Loading Cleaned Dataset...")
print("="*60)

train = pd.read_csv("outputs/train_cleaned.csv")

print("Dataset Loaded Successfully")
print("Shape :", train.shape)

# ==========================================
# Features and Target
# ==========================================

X = train.drop(["id", "health_condition"], axis=1)

y = train["health_condition"]

print("\nFeatures :", X.shape)
print("Target :", y.shape)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records :", len(X_train))
print("Validation Records :", len(X_valid))

# ==========================================
# Models
# ==========================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="mlogloss"
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100,50),
        max_iter=300,
        random_state=42
    ),

    "SVM": LinearSVC(
        random_state=42,
        max_iter=5000
    )

}

results = []

best_accuracy = 0

best_model = None

best_model_name = ""

# ==========================================
# Train Every Model
# ==========================================

for name, model in models.items():

    print("\n")
    print("="*60)
    print("Training :", name)
    print("="*60)

    # Train

    model.fit(X_train, y_train)

    # Save Model

    if name == "Decision Tree":

        filename = "decision_tree.pkl"

    elif name == "Random Forest":

        filename = "random_forest.pkl"

    elif name == "XGBoost":

        filename = "xgboost.pkl"

    elif name == "Neural Network":

        filename = "neural_network.pkl"

    elif name == "SVM":

        filename = "svm.pkl"

    joblib.dump(
        model,
        os.path.join("models", filename)
    )

    print("Saved :", filename)

    # Prediction

    prediction = model.predict(X_valid)

    # Metrics

    accuracy = accuracy_score(
        y_valid,
        prediction
    )

    precision = precision_score(
        y_valid,
        prediction,
        average="weighted"
    )

    recall = recall_score(
        y_valid,
        prediction,
        average="weighted"
    )

    f1 = f1_score(
        y_valid,
        prediction,
        average="weighted"
    )

    print("\nAccuracy :", round(accuracy,4))
    print("Precision :", round(precision,4))
    print("Recall :", round(recall,4))
    print("F1 Score :", round(f1,4))

    print("\nConfusion Matrix")

    print(confusion_matrix(
        y_valid,
        prediction
    ))

    print("\nClassification Report")

    print(classification_report(
        y_valid,
        prediction
    ))

    results.append([

        name,

        round(accuracy,4),

        round(precision,4),

        round(recall,4),

        round(f1,4)

    ])

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# ==========================================
# Save Best Model
# ==========================================

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\n")
print("="*60)
print("BEST MODEL")
print("="*60)

print("Best Model :", best_model_name)
print("Accuracy :", round(best_accuracy,4))

# ==========================================
# Save Results
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("\nModel Comparison")
print(results_df)

print("\nmodel_results.csv Saved")

print("\nTraining Completed Successfully")
print("="*60)