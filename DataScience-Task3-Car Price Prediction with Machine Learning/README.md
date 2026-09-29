# DataScience-Task3-Car Price Prediction with Machine Learning

## 🚗 Car Price Prediction with Machine Learning

This project predicts the selling price of used cars using Machine Learning. It covers the complete workflow from data cleaning and exploratory data analysis to model training, evaluation, and an interactive Streamlit web application.

## 📌 Project Overview

The project uses a used-car dataset to analyze the factors affecting car prices and build regression models to predict the selling price of a vehicle.

### Key steps covered:

- Data cleaning and preprocessing
- Handling missing values and duplicates
- Standardizing categorical values
- Feature engineering
- Exploratory Data Analysis (EDA)
- Categorical feature encoding
- Correlation analysis
- Training multiple regression models
- Model performance evaluation
- Feature importance analysis
- Interactive car price prediction

## 🔧 Data Preprocessing

The dataset is cleaned and prepared before training the models.

The preprocessing includes:

- Checking and handling null values
- Removing duplicate records
- Cleaning inconsistent categorical values
- Converting categorical values into a consistent format

## ⚙️ Feature Engineering

Additional features are created to improve the prediction process:

- **Car Age** – calculated from the vehicle's manufacturing year
- **Brand** – extracted from the car name

## 📊 Exploratory Data Analysis

The project uses visualizations to understand patterns and relationships within the dataset.

The analysis includes:

- Selling price distribution
- Fuel type vs selling price
- Car age vs selling price
- Correlation heatmap
- Feature importance visualization

## 🧠 Machine Learning Models

Three regression models are trained and compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

## 📈 Model Evaluation

The models are evaluated using the following metrics:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

The models are compared based on their performance on the test dataset.

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application called:

### AutoPulse AI | Car Price Prediction

The application provides:

- Project overview
- Interactive car price prediction
- Model performance comparison
- Feature importance visualization
- Dataset explorer

Users can enter vehicle details such as:

- Brand
- Year
- Present price
- Kilometres driven
- Fuel type
- Seller type
- Transmission
- Previous owners

The application then generates an estimated selling price.

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**
- **Streamlit**

## 📁 Project Structure


DataScience-Task3-Car Price Prediction with Machine Learning/
│
├── app.py
├── car data.csv
├── Gemini_Generated_Image_klofnvklofnvklof.png
├── requirements.txt
└── README.md
