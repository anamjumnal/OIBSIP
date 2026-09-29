# 🚗 Car Price Prediction with Machine Learning

This project predicts the selling price of used cars using Machine Learning. It covers the complete workflow from data cleaning and exploratory data analysis to model training, evaluation, and an interactive Streamlit web application.

## 📌 Project Overview

The project uses a used-car dataset to analyze the factors affecting car prices and build regression models to predict the selling price of a vehicle.

**Key steps covered:**

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

The dataset is cleaned and prepared before training the models:

- Checking and handling null values
- Removing duplicate records
- Cleaning inconsistent categorical values
- Converting categorical values into a consistent format

## ⚙️ Feature Engineering

Additional features are created to improve the prediction process:

- **Car Age** – calculated from the vehicle's manufacturing year
- **Brand** – extracted from the car name

## 📊 Exploratory Data Analysis

Visualizations are used to understand patterns and relationships within the dataset:

- Selling price distribution
- Fuel type vs selling price
- Car age vs selling price
- Correlation heatmap
- Feature importance visualization

## 🧠 Machine Learning Models

Three regression models are trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## 📈 Model Evaluation

The models are evaluated on the test dataset using:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit app: **AutoPulse AI | Car Price Prediction**

**Features:**

- Project overview
- Interactive car price prediction
- Model performance comparison
- Feature importance visualization
- Dataset explorer

**Input details:**

- Brand
- Year
- Present price
- Kilometres driven
- Fuel type
- Seller type
- Transmission
- Previous owners

The app then generates an estimated selling price.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Streamlit

## 📁 Project Structure

```
DataScience-Task3-Car Price Prediction with Machine Learning/
│
├── app.py
├── car data.csv
├── Gemini_Generated_Image_klofnvklofnvklof.png
├── requirements.txt
└── README.md
```

## ▶️ Run the Project Locally

**1. Clone the repository**

```bash
git clone https://github.com/anamjumnal/OIBSIP.git
```

**2. Navigate to the Task 3 folder**

```bash
cd "OIBSIP/DataScience-Task3-Car Price Prediction with Machine Learning"
```

**3. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

The app will open in your browser at: http://localhost:8501

## 🎯 Project Outcome

This project demonstrates an end-to-end Machine Learning workflow for used-car price prediction, including data preprocessing, feature engineering, exploratory analysis, model training, evaluation, and deployment through an interactive web application.

## 👩‍💻 Author

**Anam Jumnal**
GitHub: [anamjumnal](https://github.com/anamjumnal/OIBSIP)
