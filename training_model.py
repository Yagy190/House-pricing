import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("h_prices.csv")

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Numerical and Categorical columns
numerical_features = [
    "SquareFootage",
    "Bedrooms",
    "Bathrooms",
    "Age"
]

categorical_features = [
    "Location"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Predictions
predictions = pipeline.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Mean Absolute Error : ${mae:,.2f}")
print(f"R2 Score            : {r2:.2f}")

# Save Model
joblib.dump(pipeline, "house_price_model.pkl")

print("\nModel saved successfully.")