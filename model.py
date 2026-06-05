# ZeroError Diagnosis — model.py
# Defines and trains the ML model

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

def build_model(model_type="random_forest"):
    """
    Build and return the ML model.
    Options: 'random_forest', 'gradient_boosting', 'logistic_regression'
    """
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    }
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}")
    print(f"📦 Model selected: {model_type}")
    return models[model_type]

def train_model(model, X_train, y_train, tune=False):
    """Train the model. Optionally run GridSearchCV for hyperparameter tuning."""
    if tune:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10]
        }
        grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"🔧 Best params: {grid.best_params_}")
        return grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        print("✅ Model training complete.")
        return model