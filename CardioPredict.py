# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# LOAD DATA
# =========================
# Replace with your dataset path
df = pd.read_csv("heart.csv")

# =========================
# PHASE 1: UNDERSTANDING DATA
# =========================

# Shape
print("Shape of dataset:", df.shape)

# Data types
print("\nData Types:\n", df.dtypes)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# -------------------------
# VISUALIZATION
# -------------------------

# Histogram
df.hist(figsize=(12,10))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter: thalach vs chol
plt.scatter(df['thalach'], df['chol'])
plt.xlabel("Max Heart Rate")
plt.ylabel("Cholesterol")
plt.title("Thalach vs Chol")
plt.show()

# =========================
# PHASE 2: DATA CLEANING
# =========================

# Convert columns with '?' to numeric
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

# Fill missing values (median is robust for medical data)
df['ca'].fillna(df['ca'].median(), inplace=True)
df['thal'].fillna(df['thal'].median(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check shape after cleaning
print("\nShape after cleaning:", df.shape)

# =========================
# ENCODING (already numeric categories mostly)
# =========================
# No need for one-hot here because values are already encoded (0,1,2...)

# =========================
# PHASE 3: FEATURE ENGINEERING
# =========================

# Feature: cardiovascular load proxy
df['cardio_load'] = df['age'] * df['thalach']

# Feature: high risk flag
df['high_risk'] = ((df['trestbps'] > 140) & (df['oldpeak'] > 2)).astype(int)

# Visualization of new feature vs chol
plt.scatter(df['cardio_load'], df['chol'])
plt.xlabel("Cardio Load")
plt.ylabel("Cholesterol")
plt.title("Cardio Load vs Chol")
plt.show()

# =========================
# PHASE 4: LINEAR REGRESSION
# =========================

# Target variable
y = df['chol']

# Features (exclude chol)
X = df.drop(columns=['chol'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nR2 Score:", r2)
print("MAE:", mae)

# =========================
# COEFFICIENTS
# =========================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

# Sort by absolute importance
coefficients['abs_coef'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='abs_coef', ascending=False)

print("\nTop Features:\n", coefficients.head(10))