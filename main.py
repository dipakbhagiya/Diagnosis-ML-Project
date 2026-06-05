# ZeroError Diagnosis — main.py
# Entry point: loads data, trains model, evaluates results

import pandas as pd
from preprocess import load_and_preprocess
from model import build_model, train_model
from evaluate import evaluate_model
import joblib

def main():
    print("🔬 ZeroError Diagnosis — Starting Pipeline")

    # Step 1: Load & preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        filepath="/home/ankit-dubariya/Downloads/student-score-predictor/diabetes.csv"
    )

    # Step 2: Build model
    model = build_model(model_type="random_forest")

    # Step 3: Train
    trained_model = train_model(model, X_train, y_train)

    # Step 4: Evaluate
    evaluate_model(trained_model, X_test, y_test, feature_names)

    # Step 5: Save model
    joblib.dump(trained_model, "saved_model/zeroerrror_model.pkl")
    print("✅ Model saved to saved_model/zeroerrror_model.pkl")

if __name__ == "__main__":
    main()