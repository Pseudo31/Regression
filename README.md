# InsureCost Predictor — Outlier-Resilient Insurance Charges Regression

This project presents a comprehensive machine learning pipeline designed to predict individual insurance charges based on demographic and lifestyle features. It emphasizes robust preprocessing including advanced outlier detection and removal, feature engineering with categorical encoding, and feature selection via Recursive Feature Elimination (RFE) combined with linear regression modeling.

---

## 🚀 Project Overview

The goal is to build a reliable regression model that estimates insurance charges while minimizing noise caused by extreme outliers. This is achieved by applying the Interquartile Range (IQR) method to remove outliers from key numeric features such as age and BMI. The model pipeline then encodes categorical variables, scales features, and performs backward feature elimination to optimize predictive performance.

---

## 🔍 Features and Methodology

- **Outlier Detection & Removal:**  
  Utilizes a customizable IQR multiplier to filter outliers in numeric columns, improving model robustness.

- **Exploratory Data Analysis (EDA):**  
  Generates pair plots and correlation heatmaps before and after outlier removal for visual insight into feature relationships.

- **Categorical Feature Encoding:**  
  Applies one-hot encoding to multi-category features (`region`) and label encoding for binary categories (`sex`, `smoker`).

- **Data Splitting & Scaling:**  
  Splits data into training and testing sets (95% train, 5% test) and applies standard scaling to normalize feature distributions.

- **Feature Selection using Recursive Feature Elimination (RFE):**  
  Iteratively selects the most predictive features while training a linear regression model, evaluating performance by Root Mean Squared Error (RMSE) and R² score on test data.

---

## 📈 Model Performance

The model’s performance is evaluated across different numbers of selected features. For each iteration, it outputs:

- The subset of features selected by RFE  
- Test R² score (coefficient of determination)  
- Root Mean Squared Error (RMSE) on test data

This iterative approach provides insight into the trade-off between model complexity and predictive accuracy.

---

📊 Visualizations

- The script automatically generates:
- Pairplots to visualize feature distributions and relationships
- Correlation heatmaps highlighting important linear dependencies
- These visualizations are displayed before and after outlier removal to illustrate data cleaning effects.

🔧 Technologies Used

- Python 3.x
- Pandas, NumPy (data manipulation)
- Seaborn, Matplotlib (data visualization)
- Scikit-learn (modeling, preprocessing, feature selection)

---
