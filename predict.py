# ==========================================
# Computational Intelligence Assignment
# Generate Kaggle Submission
# ==========================================

import pandas as pd
import joblib

# ==========================================
# Load cleaned test dataset
# ==========================================

print("=" * 60)
print("Loading Test Dataset...")
print("=" * 60)

test = pd.read_csv("outputs/test_cleaned.csv")

print("Test Dataset Loaded Successfully")
print("Shape :", test.shape)

# ==========================================
# Save IDs
# ==========================================

ids = test["id"]

# ==========================================
# Features
# ==========================================

X_test = test.drop("id", axis=1)

# ==========================================
# Load Best Model
# ==========================================

print("\nLoading Best Model...")

model = joblib.load("models/best_model.pkl")

print("Best Model Loaded Successfully")

# ==========================================
# Predict
# ==========================================

print("\nPredicting...")

predictions = model.predict(X_test)

# ==========================================
# Convert Predictions to Labels
# ==========================================

# Change these if your label encoding was different.
label_map = {
    0: "at-risk",
    1: "fit",
    2: "unhealthy"
}

# If the model already returns strings, don't map them.
if isinstance(predictions[0], str):
    predicted_labels = predictions
else:
    predicted_labels = [label_map[p] for p in predictions]

# ==========================================
# Create Submission
# ==========================================

submission = pd.DataFrame({
    "id": ids,
    "health_condition": predicted_labels
})

# ==========================================
# Save CSV
# ==========================================

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully!")

print(submission.head())

print("\nTotal Predictions :", len(submission))