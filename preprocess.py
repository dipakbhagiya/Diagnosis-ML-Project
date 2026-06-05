# ZeroError Diagnosis — preprocess.py
# Handles data loading, cleaning, and feature engineering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess(filepath, target_col="Outcome", test_size=0.2):
    """
    Load CSV, clean data, engineer features, split into train/test.
    Compatible with Pima Indians Diabetes Dataset by default.
    """
    df = pd.read_csv(filepath)
    print(f"📂 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")

    # Replace zeros with NaN for medical cols (0 is biologically impossible)
    zero_invalid_cols = [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"
    ]
    df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)

    # Fill missing values with median per class
    for col in zero_invalid_cols:
        df[col] = df.groupby(target_col)[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Feature engineering
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)
    df["BP_Category"] = pd.cut(
        df["BloodPressure"],
        bins=[0, 60, 80, 90, 200],
        labels=[0, 1, 2, 3]
    ).astype("int64")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=42
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"✅ Preprocessed — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names