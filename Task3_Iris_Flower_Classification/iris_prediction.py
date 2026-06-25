import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load Dataset
df = pd.read_csv("IRIS.csv")

# Encode species
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Create Features and Target
X = df.drop("species", axis=1)
y = df["species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("✅ Model Trained Successfully")
# Predict on test data
y_pred = model.predict(X_test)

print("\nPredicted Classes:")
print(y_pred)
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()
import pandas as pd

importance = model.feature_importances_

feature_importance = pd.Series(
    importance,
    index=X.columns
)

feature_importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()
plt.bar(["Accuracy"], [accuracy])

plt.ylim(0,1.1)

plt.title("Model Accuracy")

plt.show()