import pandas as pd
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# New house
new_house = pd.DataFrame({
    "SquareFootage":[2100],
    "Bedrooms":[4],
    "Bathrooms":[3],
    "Age":[5],
    "Location":["Urban"]
})

price = model.predict(new_house)

print("="*40)
print("Predicted House Price")
print("="*40)
print(f"${price[0]:,.2f}")