#  House Price Prediction using Machine Learning

## Project Overview

This project builds a Machine Learning model to predict house prices based on various property features such as:

* Square Footage
* Number of Bedrooms
* Number of Bathrooms
* House Age
* Location (Urban, Suburban, Rural, etc.)

The model is trained using the **Random Forest Regression** algorithm and includes a preprocessing pipeline for handling categorical data.

---

##  Features

* Data preprocessing using Scikit-Learn Pipeline
* One-Hot Encoding for categorical features
* Random Forest Regression model
* Model evaluation using:

  * Mean Absolute Error (MAE)
  * R² Score
* Save and load trained models using Joblib
* Predict prices for new house data

---

##  Project Structure

```text
House-Price-Prediction/
│
├── training_model.py          # Train and save the model
├── prediction.py              # Predict price using saved model
├── house_price_model.pkl      # Trained model
├── h_prices.csv               # Dataset
├── README.md
```

---

##  Technologies Used

* Python 3.x
* Pandas
* Scikit-Learn
* Joblib

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
cd House-Price-Prediction
```

Install dependencies:

```bash
pip install pandas scikit-learn joblib
```

---

##  Dataset Features

| Feature       | Description             |
| ------------- | ----------------------- |
| SquareFootage | Total area of the house |
| Bedrooms      | Number of bedrooms      |
| Bathrooms     | Number of bathrooms     |
| Age           | Age of the house        |
| Location      | Location category       |
| Price         | Target house price      |

---

##  Training the Model

Run:

```bash
python training_model.py
```

The script will:

1. Load the dataset
2. Preprocess categorical features
3. Split data into training and testing sets
4. Train a Random Forest Regressor
5. Evaluate model performance
6. Save the trained model

---

##  Making Predictions

Run:

```bash
python prediction.py
```

Example input:

```python
new_house = {
    "SquareFootage": 2100,
    "Bedrooms": 4,
    "Bathrooms": 3,
    "Age": 5,
    "Location": "Urban"
}
```

Output:

```text
========================================
Predicted House Price
========================================
$XXX,XXX.XX
```

---

## 📈 Model Evaluation Metrics

The model uses:

* Mean Absolute Error (MAE)
* R² Score

These metrics help measure prediction accuracy and model performance.

---

## 🎯 Future Improvements

* Hyperparameter tuning
* Feature engineering
* Web application using Flask or Streamlit
* Model deployment on cloud platforms
* Real estate market trend analysis

---


##  Author

Developed as a Machine Learning project for learning regression, data preprocessing, model training, and prediction using Python and Scikit-Learn.
