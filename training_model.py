"""
House Price Prediction — Multi-Model Training & Evaluation
"""

import pathlib
import sys
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "h_prices.csv"
MODEL_PATH = BASE_DIR / "house_price_model.pkl"

NUMERICAL_FEATURES = ["SquareFootage", "Bedrooms", "Bathrooms", "Age"]
CATEGORICAL_FEATURES = ["Location"]
TARGET = "Price"


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ]
    )


def train_and_evaluate() -> None:
    if not CSV_PATH.exists():
        print(f"Dataset not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET])
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    print("=" * 64)
    print("  HOUSE PRICE MODEL TRAINING & COMPARISON")
    print("=" * 64)
    print(f"  Dataset : {CSV_PATH.name} ({len(df)} rows | Train: {len(X_train)} | Test: {len(X_test)})\n")
    print(f"  {'Model':<22} {'R2':>7} {'MAE':>12} {'RMSE':>12} {'CV R2':>8}")
    print("  " + "-" * 62)

    results = {}
    for name, reg in candidates.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", reg),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
        results[name] = (r2, mae, rmse, cv_r2, pipe)

    best_name = max(results, key=lambda k: results[k][0])
    for name, (r2, mae, rmse, cv_r2, _) in results.items():
        mark = " *" if name == best_name else ""
        print(f"  {name:<22} {r2:>7.4f}  ${mae:>10,.2f}  ${rmse:>10,.2f} {cv_r2:>8.4f}{mark}")

    best_r2, _, _, _, best_pipe = results[best_name]
    print(f"\n  Best Model -> {best_name} (R2 = {best_r2:.4f})")

    # Show Actual vs Predicted on test set
    test_preds = best_pipe.predict(X_test)
    print("\n  Test Set Comparison:")
    print(f"  {'Actual':>12} {'Predicted':>12} {'Error':>12}")
    print("  " + "-" * 38)
    for actual, pred in zip(y_test.values, test_preds):
        print(f"  ${actual:>10,.0f}  ${pred:>10,.0f}  ${pred - actual:>+10,.0f}")

    joblib.dump(
        {
            "pipeline": best_pipe,
            "model_name": best_name,
            "locations": sorted(df["Location"].dropna().unique().tolist()),
        },
        MODEL_PATH,
    )
    print(f"\n  Saved best model -> {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()

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
