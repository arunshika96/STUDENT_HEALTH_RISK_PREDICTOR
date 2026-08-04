# ==========================================
# Import Libraries
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Load Datasets
# ==========================================

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("Datasets loaded successfully.")

# ==========================================
# Dataset Information
# ==========================================

print("\nTraining Dataset")
print(train.head())

print("\nTest Dataset")
print(test.head())

print("\nTraining Shape :", train.shape)
print("Test Shape :", test.shape)

print("\nDataset Information")
print(train.info())

print("\nMissing Values")
print(train.isnull().sum())

print("\nTarget Distribution")
print(train["health_condition"].value_counts())

# ==========================================
# Exploratory Data Analysis (EDA)
# ==========================================

# Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train, x="health_condition")
plt.title("Health Condition Distribution")
plt.tight_layout()
plt.savefig("outputs/health_condition_distribution.png")
plt.close()

# Missing Values
missing = train.isnull().sum()
missing = missing[missing > 0]

plt.figure(figsize=(10,5))
missing.sort_values().plot(kind="bar")
plt.title("Missing Values")
plt.tight_layout()
plt.savefig("outputs/missing_values.png")
plt.close()

# Histogram
train.hist(figsize=(15,10))
plt.tight_layout()
plt.savefig("outputs/numerical_features.png")
plt.close()

# Correlation Heatmap
numeric = train.select_dtypes(include=["int64","float64"])

plt.figure(figsize=(10,8))
sns.heatmap(numeric.corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("\nEDA graphs saved successfully.")

# ==========================================
# Data Cleaning
# ==========================================

print("\nCleaning Missing Values...")

# Numerical columns
numerical_columns = train.select_dtypes(include=["int64","float64"]).columns

# Remove ID column
numerical_columns = numerical_columns.drop("id")

for col in numerical_columns:

    median = train[col].median()

    train[col] = train[col].fillna(median)

    if col in test.columns:
        test[col] = test[col].fillna(median)

# Categorical columns
categorical_columns = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender"
]

for col in categorical_columns:

    mode = train[col].mode()[0]

    train[col] = train[col].fillna(mode)

    test[col] = test[col].fillna(mode)

print("Missing values cleaned successfully.")

# ==========================================
# Encode Categorical Features
# ==========================================

print("\nEncoding Categorical Columns...")

encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    combined = pd.concat([train[col], test[col]])

    encoder.fit(combined)

    train[col] = encoder.transform(train[col])

    test[col] = encoder.transform(test[col])

    encoders[col] = encoder

###############
import joblib

joblib.dump(encoders, "models/feature_encoders.pkl")

print("Feature encoders saved successfully.")
####################

print("Categorical columns encoded.")

# ==========================================
# Encode Target Column
# ==========================================

target_encoder = LabelEncoder()

train["health_condition"] = target_encoder.fit_transform(
    train["health_condition"]
)

#########################
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("Target encoder saved successfully.")
#########################

print("\nTarget Classes")

for i, label in enumerate(target_encoder.classes_):
    print(i, "=", label)

# ==========================================
# Check Cleaned Dataset
# ==========================================

print("\nMissing Values After Cleaning")
print(train.isnull().sum())

print("\nTraining Dataset Preview")
print(train.head())

# ==========================================
# Save Cleaned Dataset
# ==========================================

train.to_csv("outputs/train_cleaned.csv", index=False)
test.to_csv("outputs/test_cleaned.csv", index=False)

print("\nCleaned datasets saved successfully.")

print("\nProject completed successfully up to Data Preprocessing.")