# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)



# Load Dataset
df = pd.read_csv("creditcard.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Check Fraud Distribution
print("\nFraud Distribution:")
print(df["Class"].value_counts())
# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully")
# Predict
y_pred = model.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal","Fraud"],
            yticklabels=["Normal","Fraud"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind="barh",
    figsize=(10,6),
    title="Feature Importance"
)

plt.xlabel("Importance")
plt.show()
# Feature Importance Graph
importance = pd.Series(model.feature_importances_, index=X.columns)

importance.sort_values().plot(
    kind="barh",
    figsize=(10,8),
    title="Feature Importance"
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(True)
plt.show()