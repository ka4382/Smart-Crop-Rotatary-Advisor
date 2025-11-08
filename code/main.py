import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import custom_metrics

BASE = Path(__file__).parent
SAVED_DIR = BASE / "saved_models"
SAVED_DIR.mkdir(exist_ok=True)

def load_dataset(path_candidates):
    for p in path_candidates:
        p = BASE / p
        if p.exists():
            return pd.read_csv(p)
    # fallback small dataset
    df = pd.DataFrame([
        ["Tomatoes","neutral","Loamy", 24, 1000, 500, 120,100,150],
        ["Potatoes","depleting","Loamy", 22, 900, 450, 110,95,140],
        ["Lettuce","restorative","Sandy", 18, 800, 400, 80,70,90],
    ], columns=["Name","Impact","Soil_Type","Temperature","Rainfall","Light_Intensity","Nitrogen","Phosphorus","Potassium"])
    print("No dataset found — using small fallback dataframe.")
    return df

def prepare_X_y(df, label_col_candidates=("crop","Name")):
    # choose label column
    label_col = None
    for c in label_col_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]
    y = df[label_col].astype(str).values
    X = df.drop(columns=[label_col])
    # keep numeric columns only for RandomForest training
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        # if no numeric columns, create simple numeric features (index-based)
        X_num = pd.DataFrame({"idx": np.arange(len(df))})
    return X_num.values, y

def train_random_forest(X, y, mode="fast"):
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    if mode == "fast":
        n_estimators = 50
    else:
        n_estimators = 300

    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42, verbose=0)
    t0 = time.time()
    rf.fit(X_scaled, y_enc)
    t = time.time() - t0
    print(f"Trained RandomForest ({n_estimators} trees) in {t:.1f}s")
    return rf, scaler, le

def save_objects(rf, scaler, le):
    with open(SAVED_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(SAVED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(SAVED_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"Saved model and preprocessors to: {SAVED_DIR}")

def plot_cm(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def train_model():
    """Train RandomForest model with proper feature encoding"""
    # Load data
    df = pd.read_csv(BASE / "soil.impact.csv")
    
    # Prepare numerical features
    num_features = ["Temperature", "Rainfall", "Light_Intensity", 
                   "Nitrogen", "Phosphorus", "Potassium"]
    X_num = df[num_features].values
    
    # Encode categorical features
    cat_features = ["Season", "Soil_Type", "Impact", "Fertility"]
    encoders = {}
    X_cat = []
    
    for col in cat_features:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        X_cat.append(encoded.reshape(-1, 1))
        encoders[col] = le
    
    # Combine features
    X = np.hstack([X_num] + X_cat)
    
    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df['Name'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)
    
    # Save artifacts
    with open(SAVED_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(SAVED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(SAVED_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)
    with open(SAVED_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
        
    # Save metadata
    metadata = {
        "feature_cols": num_features + cat_features,
        "classes": list(target_encoder.classes_),
        "accuracy": float(accuracy),
        "best_model": "RandomForest"
    }
    with open(SAVED_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"Training complete. Accuracy: {accuracy:.2f}")
    return rf, scaler, target_encoder, encoders, metadata

if __name__ == "__main__":
    train_model()

