"""
House Price Prediction — Interactive CLI
"""

import pathlib
import sys

import joblib
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"


def ask_number(prompt: str, default: float, cast=float, min_val: float = 0.0):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return cast(default)
        try:
            val = cast(raw)
            if val < min_val:
                print(f"  Please enter a value >= {min_val}.")
                continue
            return val
        except ValueError:
            print("  Please enter a valid number.")


def ask_location(valid_locations: list[str], default: str = "Urban") -> str:
    options = "/".join(valid_locations)
    while True:
        raw = input(f"  Location ({options}) [{default}]: ").strip()
        if not raw:
            return default
        for loc in valid_locations:
            if raw.lower() == loc.lower():
                return loc
        print(f"  Please choose one of: {options}")


def load_model():
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        print("Run 'python training_model.py' first to train and save the model.")
        sys.exit(1)

    saved = joblib.load(MODEL_PATH)
    # Support both dict bundle and legacy bare pipeline
    if isinstance(saved, dict):
        return saved["pipeline"], saved.get("model_name", "Saved Model"), saved.get(
            "locations", ["Urban", "Suburban", "Rural"]
        )
    return saved, "Random Forest", ["Urban", "Suburban", "Rural"]


def main():
    model, model_name, locations = load_model()

    print("=" * 52)
    print(f"  HOUSE PRICE PREDICTOR  ({model_name})")
    print("=" * 52)
    print("  Press Enter to accept the default value in [ ]\n")

    while True:
        sqft = ask_number("Square Footage", 2100, cast=float, min_val=100)
        beds = ask_number("Bedrooms", 4, cast=int, min_val=1)
        baths = ask_number("Bathrooms", 3, cast=int, min_val=1)
        age = ask_number("House Age (years)", 5, cast=int, min_val=0)
        location = ask_location(locations, default=locations[0] if "Urban" not in locations else "Urban")

        house_df = pd.DataFrame([{
            "SquareFootage": sqft,
            "Bedrooms": beds,
            "Bathrooms": baths,
            "Age": age,
            "Location": location,
        }])

        price = model.predict(house_df)[0]
        print("\n" + "-" * 52)
        print(f"  Predicted House Price : ${price:,.2f}")
        print("-" * 52)

        again = input("\n  Predict another house? (y/N): ").strip().lower()
        if again not in ("y", "yes"):
            print("\n  Goodbye!")
            break
        print()


if __name__ == "__main__":
    main()
