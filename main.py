import pandas as pd
import numpy as np
import os
import cv2
import pickle
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/final_multimodal_dataset.csv"
MODEL_PATH = "models/best_model.pkl"

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})
    return df

# -----------------------------
# TEXT FEATURES
# -----------------------------
def process_text(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    text_features = vectorizer.fit_transform(texts)
    return text_features.toarray(), vectorizer

# -----------------------------
# IMAGE FEATURES
# -----------------------------
def extract_image_features(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        img = cv2.imdecode(
            np.frombuffer(response.content, np.uint8),
            cv2.IMREAD_COLOR
        )
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        return img.flatten()
    except:
        return np.zeros(64 * 64 * 3)

def process_images(image_urls):
    return np.array([extract_image_features(url) for url in image_urls])

# -----------------------------
# FEATURE FUSION
# -----------------------------
def combine_features(text_feat, image_feat):
    return np.hstack((text_feat, image_feat))

# -----------------------------
# MODEL TRAINING
# -----------------------------
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n{name} Accuracy: {acc}")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm)
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        plt.show()

    return results

# -----------------------------
# PLOT COMPARISON
# -----------------------------
def plot_results(results):
    names = list(results.keys())
    values = list(results.values())

    plt.figure()
    plt.bar(names, values)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()

# -----------------------------
# SAVE BEST MODEL
# -----------------------------
def save_best_model(models, results, vectorizer):
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    if not os.path.exists("models"):
        os.makedirs("models")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((best_model, vectorizer), f)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("Loading dataset...")
    df = load_data()

    print("Processing text...")
    text_features, vectorizer = process_text(df['text'])

    print("Processing images...")
    image_features = process_images(df['image_url'])

    print("Combining features...")
    X = combine_features(text_features, image_features)
    y = df['label']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training models...")
    models = train_models(X_train, y_train)

    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)

    print("Plotting results...")
    plot_results(results)

    print("Saving best model...")
    save_best_model(models, results, vectorizer)


if __name__ == "__main__":
    main()