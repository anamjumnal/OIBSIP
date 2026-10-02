# 📈 Sales Prediction Using Python

**Data Science · Task 5 · Oasis Infobyte Internship**

A regression project that predicts product sales from advertising spend on **TV**, **Radio** and **Newspaper**, with an interactive website that shows the results.
---

## 📌 Overview

Businesses need to know which advertising channels actually drive sales. This project builds and compares three regression models on the classic Advertising dataset (200 markets) to predict sales and measure each channel's impact.

## 🗂️ Dataset

`Advertising.csv`: 200 rows, 4 columns, no missing values.

| Column | Description |
|---|---|
| TV | Advertising spend on TV ($ thousands) |
| Radio | Advertising spend on Radio ($ thousands) |
| Newspaper | Advertising spend on Newspaper ($ thousands) |
| Sales | Units sold (thousands) |

## 🔍 What the project covers

- Data loading, null check and descriptive statistics
- Pairplot of all features
- Individual scatter plots: Sales vs TV, Radio and Newspaper
- Correlation matrix heatmap
- 80/20 train/test split
- Models: Linear Regression (baseline), Polynomial Regression (degree 2) and Random Forest Regressor
- Evaluation with MAE, RMSE and R²
- Residual analysis of the best model
- Feature importance and coefficient interpretation

## 📊 Results

Test set performance (20% hold-out, `random_state=42`):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 1.461 | 1.782 | 0.899 |
| **Polynomial (degree 2)** | **0.526** | **0.643** | **0.987** |
| Random Forest | 0.613 | 0.740 | 0.983 |

**Best model:** Polynomial Regression (degree 2). The residuals scatter randomly around zero, so the errors are not systematic.

## 💡 Key Findings

- **TV** has the highest overall impact on sales (63% Random Forest importance).
- **Radio** gives the most sales per dollar spent (largest linear coefficient, about 0.19 per $1k).
- **Newspaper** has almost no effect once TV and Radio are included.
- Adding polynomial terms captures the interaction between channels and raises R² from 0.90 to 0.99.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook · HTML/CSS/JavaScript

## 📁 Project Structure

```
DataScience-Task5-SalesPredictionUsingPython/
├── index.html               # Interactive website
├── Sales_Prediction.ipynb   # Full analysis notebook
├── Advertising.csv          # Dataset
└── README.md
```

## ▶️ Run Locally

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Sales_Prediction.ipynb
```

Keep `Advertising.csv` in the same folder as the notebook. To view the website, open `index.html` in any browser.

## 👤 Author

**Anam Jumnal**
Data Science Intern, Oasis Infobyte
