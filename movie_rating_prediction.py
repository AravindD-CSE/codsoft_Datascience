import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

df = df.dropna()

le = LabelEncoder()

df["Genre"] = le.fit_transform(df["Genre"])
df["Director"] = le.fit_transform(df["Director"])
df["Actor 1"] = le.fit_transform(df["Actor 1"])
df["Actor 2"] = le.fit_transform(df["Actor 2"])
df["Actor 3"] = le.fit_transform(df["Actor 3"])

X = df[["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted Ratings:")
print(predictions[:10])
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)