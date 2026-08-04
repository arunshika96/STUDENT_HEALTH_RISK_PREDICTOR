# ==========================================
# Student Health Risk Prediction - Web App
# CIS 6005 Computational Intelligence Assignment
# ==========================================
#
# This Flask app loads the trained model (models/best_model.pkl,
# an XGBoost classifier) and serves a form where a user enters
# lifestyle/health measurements and receives a predicted health
# risk category: at-risk, fit, or unhealthy.
#
# IMPORTANT: The encoding below must exactly match the encoding
# used in model_training.py / your data-cleaning step, otherwise
# predictions will be meaningless. This app uses:
#   - Alphabetical LabelEncoder-style integer codes for categorical
#     fields (confirmed against your saved models — see report).
#   - Same 13-feature column order used during training.

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ------------------------------------------
# Load trained model
# ------------------------------------------
# best_model.pkl = the best-performing model from training (XGBoost).
# Swap the filename below to try a different saved model, e.g.
# "models/decision_tree.pkl" or "models/neural_network.pkl".
MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH)

# ------------------------------------------
# Feature order — MUST match training exactly
# ------------------------------------------
FEATURE_ORDER = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

# ------------------------------------------
# Categorical encodings (alphabetical order,
# matching sklearn's default LabelEncoder)
# ------------------------------------------
DIET_TYPE_MAP = {"balanced": 0, "non-veg": 1, "veg": 2}
STRESS_LEVEL_MAP = {"high": 0, "low": 1, "medium": 2}
SLEEP_QUALITY_MAP = {"average": 0, "good": 1, "poor": 2}
PHYSICAL_ACTIVITY_MAP = {"active": 0, "moderate": 1, "sedentary": 2}
SMOKING_ALCOHOL_MAP = {"no": 0, "occasional": 1, "yes": 2}
GENDER_MAP = {"female": 0, "male": 1, "other": 2}

# Target label decoding (alphabetical order): 0=at-risk, 1=fit, 2=unhealthy
TARGET_MAP = {0: "at-risk", 1: "fit", 2: "unhealthy"}

TARGET_DESCRIPTIONS = {
    "at-risk": "Some lifestyle indicators suggest elevated health risk. Consider reviewing sleep, activity, and stress levels.",
    "fit": "Lifestyle indicators are broadly consistent with good health.",
    "unhealthy": "Multiple lifestyle indicators suggest significant health risk. Consulting a healthcare professional is advised.",
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    try:
        # Numeric fields
        sleep_duration = float(form["sleep_duration"])
        heart_rate = float(form["heart_rate"])
        bmi = float(form["bmi"])
        calorie_expenditure = float(form["calorie_expenditure"])
        step_count = float(form["step_count"])
        exercise_duration = float(form["exercise_duration"])
        water_intake = float(form["water_intake"])

        # Categorical fields -> encode
        diet_type = DIET_TYPE_MAP[form["diet_type"]]
        stress_level = STRESS_LEVEL_MAP[form["stress_level"]]
        sleep_quality = SLEEP_QUALITY_MAP[form["sleep_quality"]]
        physical_activity_level = PHYSICAL_ACTIVITY_MAP[form["physical_activity_level"]]
        smoking_alcohol = SMOKING_ALCOHOL_MAP[form["smoking_alcohol"]]
        gender = GENDER_MAP[form["gender"]]

    except (KeyError, ValueError) as e:
        return render_template("index.html", error=f"Invalid input: {e}")

    row = {
        "sleep_duration": sleep_duration,
        "heart_rate": heart_rate,
        "bmi": bmi,
        "calorie_expenditure": calorie_expenditure,
        "step_count": step_count,
        "exercise_duration": exercise_duration,
        "water_intake": water_intake,
        "diet_type": diet_type,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "physical_activity_level": physical_activity_level,
        "smoking_alcohol": smoking_alcohol,
        "gender": gender,
    }
    features = pd.DataFrame([row], columns=FEATURE_ORDER)

    prediction_code = int(model.predict(features)[0])
    prediction_label = TARGET_MAP[prediction_code]

    # Probability breakdown, if the model supports it
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        probabilities = {
            TARGET_MAP[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)
        }

    return render_template(
        "index.html",
        prediction=prediction_label,
        description=TARGET_DESCRIPTIONS[prediction_label],
        probabilities=probabilities,
        form_values=form,
    )


if __name__ == "__main__":
    app.run(debug=True)
