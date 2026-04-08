# Airbnb ML Dashboard

Interactive Machine Learning dashboard for the **NYC Airbnb dataset** built using **Streamlit**.

## Features

* Data ingestion and preview
* Exploratory Data Analysis (EDA)
* Column selection and preprocessing
* Outlier removal (Z-score + IQR)
* Model training

  * Random Forest
  * KNN
  * Linear Regression
* Automatic classification / regression detection
* Performance evaluation tab
* R², MSE, RMSE, Accuracy
* Actual vs Predicted comparison graph

---

## Dataset

Designed for:

```
AB_NYC_2019.csv
```

But works with any structured dataset.

---

## Installation

Clone the repository

```
git clone https://github.com/YOUR_USERNAME/ml-airbnb-dashboard.git
cd ml-airbnb-dashboard
```

Install dependencies

```
pip install -r requirements.txt
```

Run the app

```
streamlit run app.py
```

---

## Models Supported

### Classification

* Random Forest Classifier
* KNN Classifier
* Logistic Regression

### Regression

* Random Forest Regressor
* Linear Regression

---

## Preprocessing

* Drop columns
* Handle missing values
* Z-score outlier removal
* IQR outlier removal
* Feature scaling
* One-hot encoding

---

## Performance Metrics

Regression:

* R² Score
* MSE
* RMSE

Classification:

* Accuracy
* Classification Report

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Plotly
* Pandas
* NumPy

---

## Author

Suryansh Dwivedi
