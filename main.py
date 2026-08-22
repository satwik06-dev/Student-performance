import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# 1. Load Dataset
df = pd.read_csv("C:/Users/Sekha/Downloads/StudentsPerformance_3_lyst1729690388778.csv")

# Optional: Create an aggregate total score target variable
df['total score'] = df['math score'] + df['reading score'] + df['writing score']

# 2. Input and Target Split
X = df.drop(columns=['math score', 'total score'], errors='ignore')
y = df['math score']

# 3. Train-Test Split (Perform BEFORE preprocessing to avoid data leakage)[cite: 1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Encoding Mapping
nominal_cols = ['gender', 'race/ethnicity', 'lunch']
ordinal_cols = ['parental level of education', 'test preparation course']

# Define explicit ordering for ordinal categories
edu_order = [
    'some high school', 'high school', 'some college', 
    "associate's degree", "bachelor's degree", "master's degree"
]
prep_order = ['none', 'completed']

# 5. Build Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('nom', OneHotEncoder(drop='first'), nominal_cols),
        ('ord', OrdinalEncoder(categories=[edu_order, prep_order]), ordinal_cols),
        ('num', StandardScaler(), ['reading score', 'writing score'])
    ]
)

# 6. Build & Train Model Pipeline (Switched to Random Forest for better nonlinear capture)[cite: 1]
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

# 7. Predict and Evaluate
y_pred = model_pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 8. Visual Evaluation
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Math Scores")
plt.ylabel("Predicted Math Scores")
plt.title("Actual vs Predicted Math Scores")
plt.show()
