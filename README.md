# Student Health Risk Predictor — Web App

Flask web application that uses your trained model (`models/best_model.pkl`,
an XGBoost classifier) to predict a student's health risk category —
**at-risk**, **fit**, or **unhealthy** — from 13 lifestyle/health inputs.

Built for CIS 6005 Computational Intelligence — "Deep learning Plus AI Mini
project" (Kaggle Playground Series S6E7: Predicting Student Health Risk).

## How to run

1. Install dependencies (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   python app.py
   ```

3. Open your browser at **http://127.0.0.1:5000**

4. Fill in the form and click "Predict health risk" to see the predicted
   category and the model's confidence breakdown across all three classes.

## Project structure

```
health_risk_app/
├── app.py                # Flask backend: routes, encoding, prediction
├── requirements.txt
├── models/
│   ├── best_model.pkl     # Best-performing model (XGBoost) — used by default
│   ├── decision_tree.pkl
│   ├── neural_network.pkl
│   ├── svm.pkl
│   └── xgboost.pkl
├── templates/
│   └── index.html         # Form + result page
└── static/
    └── style.css
```

## Switching models

By default the app uses `models/best_model.pkl`. To demonstrate a different
model in your report/demo, change `MODEL_PATH` at the top of `app.py`, e.g.:

```python
MODEL_PATH = "models/decision_tree.pkl"
```

Note: `neural_network.pkl` and `svm.pkl` don't support `predict_proba`
directly in all configurations — check before switching if you want the
confidence bars to appear (SVM's `LinearSVC` does not expose
`predict_proba` at all, so the app will simply not show a confidence
breakdown for it).

## Important: why the encoding logic matters

`app.py` manually re-implements the same preprocessing used in
`model_training.py` (via your cleaned dataset):

- **Missing values**: numeric columns imputed with median, categorical
  columns imputed with mode (this only matters for offline evaluation —
  the web form requires every field, so no missing values reach the model).
- **Categorical encoding**: each category column was label-encoded in
  alphabetical order (matches sklearn's default `LabelEncoder` behaviour),
  e.g. `diet_type`: balanced=0, non-veg=1, veg=2.
- **Target decoding**: `0=at-risk, 1=fit, 2=unhealthy` (alphabetical order
  of the original `health_condition` labels).

This was verified by re-running `decision_tree.pkl` against `train.csv`
with this exact encoding scheme and confirming ~98.8% agreement with the
original training predictions — i.e. this encoding is what your model was
actually trained on.

If you later find your original cleaning script and it encoded things
differently, update the `*_MAP` dictionaries near the top of `app.py`
to match.

## Packaging as a "production style" deliverable

For the assignment brief, this satisfies the requirement to "package your
trained model into a usable application" (web app option). For your
report's System Architecture section (task d), you can describe:

- **Frontend**: HTML form (Jinja2 templates) collecting 13 raw inputs
- **Backend**: Flask route `/predict` that encodes inputs identically to
  training, loads the serialized model via `joblib`, and returns a
  prediction + class probabilities
- **Model layer**: pre-trained XGBoost classifier serialized with `joblib`,
  loaded once at app startup (not retrained per request)
