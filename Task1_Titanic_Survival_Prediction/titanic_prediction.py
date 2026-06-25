import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("train.csv")

# ==============================
# Handle Missing Values
# ==============================

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# ==============================
# Remove Unnecessary Columns
# ==============================

df = df.drop("Cabin", axis=1)

# ==============================
# Encode Categorical Data
# ==============================

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# ==============================
# Drop Unwanted Columns
# ==============================

df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

# ==============================
# Features and Target
# ==============================

X = df.drop("Survived", axis=1)
y = df["Survived"]

# ==============================
# Split Dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==============================
# Train Model
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==============================
# Prediction
# ==============================

y_pred = model.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])

# ==============================
# Accuracy
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy,4))

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==============================
# Classification Report
# ==============================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===================================================
# GRAPH 1 : Confusion Matrix
# ===================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived","Survived"]
)

disp.plot(cmap="Blues")

plt.title("Titanic Survival Confusion Matrix")

plt.show()

# ===================================================
# GRAPH 2 : Actual vs Predicted
# ===================================================

plt.figure(figsize=(10,5))

plt.plot(
    y_test.values[:30],
    marker='o',
    label="Actual"
)

plt.plot(
    y_pred[:30],
    marker='x',
    label="Predicted"
)

plt.title("Actual vs Predicted Survival")

plt.xlabel("Passengers")

plt.ylabel("Survival")

plt.legend()

plt.grid(True)

plt.show()