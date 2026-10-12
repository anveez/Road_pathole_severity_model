import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("==================================================")
print("   ROAD POTHOLE SEVERITY CLASSIFICATION MODEL   ")
print("==================================================")

df = pd.read_csv("potholes.csv")
print(f"\n[INFO] Dataset loaded successfully! Shape: {df.shape}")
print("\nFirst 5 rows of data:")
print(df.head())

# ==========================================
# 2. DATA PREPROCESSING & FEATURE ENCODING
# ==========================================
print("\n[INFO] Preprocessing data...")

# Label Encoders for categorical features
road_type_mapping = {
    "Residential Road": 0,
    "Collector Street": 1,
    "Urban Arterial": 2,
    "Highway / Expressway": 3
}

water_mapping = {
    "Dry": 0,
    "Damp": 1,
    "Standing Water / Flooded": 2
}

severity_order = ["Low", "Moderate", "High", "Critical"]
severity_mapping = {label: idx for idx, label in enumerate(severity_order)}
reverse_severity_mapping = {idx: label for label, idx in severity_mapping.items()}

# Apply mapping
df_clean = df.copy()
df_clean["road_type_encoded"] = df_clean["road_type"].map(road_type_mapping)
df_clean["water_presence_encoded"] = df_clean["water_presence"].map(water_mapping)
df_clean["severity_encoded"] = df_clean["severity_level"].map(severity_mapping)

# Features and Target
feature_cols = [
    "pothole_depth_cm",
    "pothole_diameter_cm",
    "surface_area_sq_cm",
    "road_type_encoded",
    "traffic_speed_kmh",
    "traffic_volume_daily",
    "water_presence_encoded",
    "vibration_peak_g"
]

X = df_clean[feature_cols]
y = df_clean["severity_encoded"]

# Split into Train and Test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("\n[INFO] Training Machine Learning Models...")

# Model 1: Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=8, min_samples_split=4, random_state=42)
dt_model.fit(X_train, y_train)

# Model 2: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

# Model 3: Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n" + "="*50)
print("                 MODEL EVALUATION                 ")
print("="*50)

# Evaluate Decision Tree
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
dt_f1 = f1_score(y_test, dt_preds, average="weighted")
print(f"\n[DECISION TREE]     Accuracy : {dt_acc * 100:.2f}% | F1-Score: {dt_f1:.4f}")

# Evaluate Random Forest
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, average="weighted")
print(f"[RANDOM FOREST]     Accuracy : {rf_acc * 100:.2f}% | F1-Score: {rf_f1:.4f}")

# Evaluate Gradient Boosting
gb_preds = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_preds)
gb_f1 = f1_score(y_test, gb_preds, average="weighted")
print(f"[GRADIENT BOOSTING] Accuracy : {gb_acc * 100:.2f}% | F1-Score: {gb_f1:.4f}")

# Select the champion model
candidate_models = [
    ("Random Forest", rf_model, rf_acc, rf_f1, rf_preds),
    ("Gradient Boosting", gb_model, gb_acc, gb_f1, gb_preds),
    ("Decision Tree", dt_model, dt_acc, dt_f1, dt_preds)
]
best_name, best_model, best_acc, best_f1, best_preds = max(candidate_models, key=lambda m: m[2])

print(f"\n[CHAMPION] Model Selected: {best_name} with {best_acc * 100:.2f}% Accuracy!")

print(f"\n--- Detailed Classification Report ({best_name}) ---")
target_names = [severity_order[i] for i in range(len(severity_order))]
print(classification_report(y_test, best_preds, target_names=target_names))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, best_preds))

# Feature Importance
print("\n--- Feature Importance Breakdown ---")
feature_importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in feature_importances.items():
    print(f"  {feat:<25}: {imp * 100:.2f}%")

# ==========================================
# 5. SAVE MODEL & ARTIFACTS FOR STREAMLIT APP
# ==========================================
artifacts = {
    "model": best_model,
    "model_name": best_name,
    "dt_model": dt_model,
    "rf_model": rf_model,
    "gb_model": gb_model,
    "feature_cols": feature_cols,
    "road_type_mapping": road_type_mapping,
    "water_mapping": water_mapping,
    "severity_order": severity_order,
    "severity_mapping": severity_mapping,
    "reverse_severity_mapping": reverse_severity_mapping,
    "metrics": {
        "accuracy": float(best_acc),
        "f1_score": float(best_f1),
        "dt_accuracy": float(dt_acc),
        "rf_accuracy": float(rf_acc),
        "gb_accuracy": float(gb_acc)
    }
}

joblib.dump(artifacts, "pothole_model_bundle.pkl")
print(f"\n[SUCCESS] Champion Model bundle ({best_name}) saved as 'pothole_model_bundle.pkl'!")
