import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset
df = pd.read_csv("C:/Users/Sekha/Downloads/StudentsPerformance_3_lyst1729690388778.csv")

# 2. Exploratory Data Analysis (EDA)
print(df.head())
print(df.shape)
print(df.info())
print("Missing values:\n", df.isnull().sum())
print(df.describe())

# Data Visualizations
sns.countplot(x='gender', data=df)
plt.show()

sns.histplot(df['math score'], kde=True)
plt.show()

sns.boxplot(x='gender', y='math score', data=df)
plt.show()

# 3. Data Preprocessing
le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['lunch'] = le.fit_transform(df['lunch'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])

# 4. Input and Output Split
X = df.drop('math score', axis=1)
y = df['math score']

# 5. Split Dataset into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict Results
y_pred = model.predict(X_test)

# 8. Check Accuracy
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 9. Plot Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Math Scores")
plt.show()
