# ML/train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def train_model(data_path="dataset_preprocessed.csv", model_path="ML/foundation_model.pkl"):
    # Load dataset
    df = pd.read_csv(data_path)

    # Select features and label
    X = df[["R", "G", "B"]]
    y = df["shade_name"]

    # Normalize RGB features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Nearest Neighbors model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_scaled, y)

    # Save everything neatly as a dictionary
    model_data = {
        "scaler": scaler,
        "model": model,
        "data": df
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    print(f"✅ Model trained and saved successfully at {model_path}")

if __name__ == "__main__":
    train_model()
