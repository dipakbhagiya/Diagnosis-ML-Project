# ZeroError Diagnosis
### ML-powered Healthcare Disease Prediction System

## 📁 Project Structure
zeroerrror-diagnosis/
├── data/
│   └── diabetes.csv          # Pima Indians Diabetes Dataset
├── saved_model/
│   └── zeroerrror_model.pkl
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── main.py
├── model.py
├── preprocess.py
├── evaluate.py
└── requirements.txt

## 🚀 Quick Start

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Save as: data/diabetes.csv

# 3. Run the project
python main.py

## 📊 Expected Results
- Accuracy  : ~78–82%
- F1 Score  : ~74–79%
- ROC-AUC   : ~83–88%

## 🔧 Model Options
Change model_type in main.py:
- "random_forest"       ← default (best)
- "gradient_boosting"   ← slower, slightly better
- "logistic_regression" ← fast baseline

## 📦 Dataset
Pima Indians Diabetes Dataset (UCI)
768 samples | 8 features | Binary classification
Target: Outcome (1 = Diabetic, 0 = Not Diabetic)

## ✅ Features Used
- Pregnancies, Glucose, BloodPressure
- SkinThickness, Insulin, BMI
- DiabetesPedigreeFunction, Age
- Engineered: BMI_Age, Glucose_Insulin_Ratio, BP_Category