from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "278k_song_labelled.csv"
MODEL_FILE = MODELS_DIR / "mood_pipeline.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load dataset
# =========================================================
print(f"Loading dataset from: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

print(f"Original dataset shape: {df.shape}")
print("Columns:", list(df.columns))


# =========================================================
# Clean dataset
# =========================================================
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.drop_duplicates()

target_col = "labels"

feature_cols = [
    "duration (ms)",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "spec_rate"
]

missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing expected feature columns: {missing_features}")

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

df = df.dropna(subset=[target_col])

X = df[feature_cols]
y = df[target_col].astype(int)

print(f"Cleaned dataset shape: {df.shape}")
print(f"Feature columns used: {feature_cols}")
print(f"Target column: {target_col}")
print(f"Unique labels found: {sorted(y.unique().tolist())}")


# =========================================================
# label-to-mood mapping
# =========================================================
label_to_mood = {
    0: "sad",
    1: "happy",
    2: "energetic",
    3: "calm"
}
mood_to_label = {
    "sad": 0,
    "happy": 1,
    "energetic": 2,
    "calm": 3
}

# =========================================================
# Split data
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# =========================================================
# Build pipeline
# =========================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])


# =========================================================
# Train model
# =========================================================
print("Training RandomForest model...")
pipeline.fit(X_train, y_train)


# =========================================================
# Evaluate model
# =========================================================
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(cm)


# Feature importance
classifier = pipeline.named_steps["classifier"]
importances = classifier.feature_importances_

feature_importance = dict(
    sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )
)

print("\nFeature Importances:")
for feature, score in feature_importance.items():
    print(f"{feature}: {score:.4f}")

# =========================================================
# Save model and metadata
# =========================================================
joblib.dump(pipeline, MODEL_FILE)

metadata = {
    "data_file": DATA_FILE.name,
    "target_column": target_col,
    "feature_columns": feature_cols,
    "classes": sorted(y.unique().tolist()),
    "label_to_mood": {
        0: "sad",
        1: "happy",
        2: "energetic",
        3: "calm"
    },
    "mood_to_label": {
        "sad": 0,
        "happy": 1,
        "energetic": 2,
        "calm": 3
    },
    "model_file": MODEL_FILE.name,
    "model_type": "RandomForestClassifier",
    "test_size": 0.2,
    "random_state": 42
}
metadata["feature_importance"] = feature_importance

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nSaved files:")
print(f"- Model pipeline: {MODEL_FILE}")
print(f"- Metadata: {METADATA_FILE}")
print("\nTraining complete.")