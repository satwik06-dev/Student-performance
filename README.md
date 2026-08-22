# ==============================================================================
# PROJECT: Student Performance Prediction System using Machine Learning
# File: StudentsPerformance_3_lyst1729690388778.csv
# Tech Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------------------
# Step 1: Data Collection (Module 1: Data Input Module)
# ------------------------------------------------------------------------------
file_path = "StudentsPerformance_3_lyst1729690388778.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("STEP 1: DATA COLLECTION")
print("=" * 60)
print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(3))

# ------------------------------------------------------------------------------
# Step 2: Data Preprocessing (Module 2: Data Processing Module)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING & TARGET CREATION")
print("=" * 60)

# Check missing values
print("Missing values per column:\n", df.isnull().sum())

# Define Target Variable: Final_Result (Pass = 1, Fail/At-Risk = 0)
# Benchmark: Average score of math, reading, and writing >= 50 is Pass
df["Average_Score"] = (
    df["math score"] + df["reading score"] + df["writing score"]
) / 3
df["Final_Result"] = np.where(df["Average_Score"] >= 50, 1, 0)

print(f"\nFinal Result Distribution:")
print(f"Pass (1)     : {sum(df['Final_Result'] == 1)}")
print(f"Fail/Risk (0): {sum(df['Final_Result'] == 0)}")

# ------------------------------------------------------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)
print(
    df.groupby("Final_Result")[
        ["math score", "reading score", "writing score"]
    ].mean()
)

# ------------------------------------------------------------------------------
# Step 4: Feature Selection
# ------------------------------------------------------------------------------
# Independent features (socio-economic background, course prep, assessments)
X = df.drop(
    columns=[
        "math score",
        "reading score",
        "writing score",
        "Average_Score",
        "Final_Result",
    ]
)
y = df["Final_Result"]

nominal_cols = ["gender", "race/ethnicity", "lunch"]
ordinal_cols = ["parental level of education", "test preparation course"]

edu_levels = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]
prep_levels = ["none", "completed"]

preprocessor = ColumnTransformer(
    transformers=[
        ("nom", OneHotEncoder(drop="first"), nominal_cols),
        (
            "ord",
            OrdinalEncoder(categories=[edu_levels, prep_levels]),
            ordinal_cols,
        ),
    ]
)

# ------------------------------------------------------------------------------
# Step 5 & 6: Model Selection and Training (Module 3: Model Training Module)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 & 6: MODEL TRAINING (80% Train, 20% Test)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine (SVM)": SVC(
        kernel="linear", probability=True, random_state=42
    ),
}

trained_pipelines = {}
performance_table = []

for name, model in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", model),
        ]
    )
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    # --------------------------------------------------------------------------
    # Step 8: Model Evaluation
    # --------------------------------------------------------------------------
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    performance_table.append(
        {
            "Algorithm": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
        }
    )

eval_df = pd.DataFrame(performance_table)
print("\n--- Model Evaluation Summary Table ---")
print(eval_df.to_string(index=False))

# Confusion Matrix for Decision Tree
best_model = trained_pipelines["Decision Tree"]
y_pred_dt = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_dt)

print("\n--- Confusion Matrix (Decision Tree) ---")
print(f"Predicted Fail  Predicted Pass")
print(f"Actual Fail:  TN = {cm[0][0]}        FP = {cm[0][1]}")
print(f"Actual Pass:  FN = {cm[1][0]}        TP = {cm[1][1]}")

# ------------------------------------------------------------------------------
# Step 7: Prediction Output (Module 4: Prediction Module)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: PREDICTION ON NEW STUDENT")
print("=" * 60)

sample_student = pd.DataFrame(
    [
        {
            "gender": "female",
            "race/ethnicity": "group C",
            "parental level of education": "some high school",
            "lunch": "free/reduced",
            "test preparation course": "none",
        }
    ]
)

pred = best_model.predict(sample_student)[0]
prob = best_model.predict_proba(sample_student)[0]

print("Input Student Profile:")
print(sample_student.to_string(index=False))
print(
    f"\nPrediction: {'PASS' if pred == 1 else 'FAIL / AT-RISK (Needs Action)'}"
)
print(f"Confidence : {prob[pred]*100:.2f}%")

# ------------------------------------------------------------------------------
# Step 8 (Contd.): Visual Evaluation (Module 5: Visualization Module)
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Plot 1: Confusion Matrix Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"],
)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Plot 2: Model Accuracy Comparison
plt.subplot(1, 2, 2)
sns.barplot(x="Algorithm", y="Accuracy (%)", data=eval_df, palette="viridis")
plt.title("Model Comparison")
plt.xticks(rotation=25)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()
