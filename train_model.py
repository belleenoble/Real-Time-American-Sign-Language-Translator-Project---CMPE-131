import os 

import joblib 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


STATIC_DATA_PATH = "data/landmarks.csv"
MOTION_DATA_PATH = "data/motion_landmarks.csv"

STATIC_MODEL_PATH = "models/static_sign_classifier.joblib"
MOTION_MODEL_PATH = "models/motion_sign_classifier.joblib"

os.makedirs("models", exist_ok=True) #creats the models directory if it dosent exist

def train_classifier(data_path, model_path, sign_type):
    # Loads the dataset, trains a RandomForest classifier, evaluates it on a test split, then saves the trained model with joblib. Prints the accuracy and classification report.

    if not os.path.exists(data_path):
        print(f"Error: {sign_type} dataset not found at {data_path}. Please collect data first.")
        return

    data = pd.read_csv(data_path)

    if data.empty:
        print(f"Error: {sign_type} dataset at {data_path} is empty. Please collect data first.")
        return

    label_counts = data["label"].value_counts()
    too_few = label_counts[label_counts < 2]

    if not too_few.empty:
        print(f"Warning these {sign_type} labels have fewer than 2 samples and will be dropped: {list(too_few.index)}")

        data = data[~data["label"].isin(too_few.index)]

    feature = data.drop(columns=["label"]).values
    label = data["label"].values

    feature_train, feature_test, labels_train, labels_test = train_test_split(feature, label, test_size=0.2, random_state=42, stratify=labels)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(feature_train, labels_train)

    predictions = model.predict(feature_test)
    accuracy = accuracy_score(labels_test, predictions)

    print(f"\n=== {sign_type} sign model ===")
    print(f"Training samples: {len(labels_train)}")
    print(f"Test samples: {len(labels_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(labels_test, predictions, zero_division=0))

    joblib.dump(model, model_path)
    print(f"Trained {sign_type} sign model saved to {model_path}\n")

def predict_with_confidence(model, feature_vector):
    #Runs the model on the feature vector and returns the predictions. This is basically the confidence score for each class
    
    feature_vector = np.array(feature_vector).reshape(1, -1)
    probabilities = model.predict_proba(feature_vector)[0]
    best_class_index = np.argmax(probabilities)
    return model.classes_[best_class_index], probabilities[best_class_index]

if __name__ == "__main__":
    train_classifier(STATIC_DATA_PATH, STATIC_MODEL_PATH, "Static")
    train_classifier(MOTION_DATA_PATH, MOTION_MODEL_PATH, "Motion")