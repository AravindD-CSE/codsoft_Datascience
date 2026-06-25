import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

# Remove missing values
df = df.dropna()

# -----------------------------
# Encode Categorical Columns
# -----------------------------
le = LabelEncoder()

df["Genre"] = le.fit_transform(df["Genre"])
df["Director"] = le.fit_transform(df["Director"])
df["Actor 1"] = le.fit_transform(df["Actor 1"])
df["Actor 2"] = le.fit_transform(df["Actor 2"])
df["Actor 3"] = le.fit_transform(df["Actor 3"])

# -----------------------------
# Select Features and Target
# -----------------------------
X = df[["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]]
y = df["Rating"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# -----------------------------
# Predict Ratings
# -----------------------------
predictions = model.predict(X_test)

print("\nFirst 10 Predicted Ratings:")
print(predictions[:10])

# -----------------------------
# Calculate Error
# -----------------------------
mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error (MAE):", round(mae,3))

# =====================================================
# GRAPH 1 : Actual vs Predicted Ratings
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test,
            predictions,
            color="blue",
            alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.title("Actual vs Predicted Movie Ratings")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.grid(True)

plt.show()

# =====================================================
# GRAPH 2 : Feature Importance
# =====================================================

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

plt.bar(features, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=25)

plt.show()