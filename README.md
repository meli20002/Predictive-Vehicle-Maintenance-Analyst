# Predictive Vehicle Maintenance Analyst

A data science project analyzing sensor data to predict vehicle component failures. This project showcases the full machine learning pipeline from data cleaning to model deployment, demonstrating expertise in predictive maintenance applications.

## Project Overview

This project uses the AI4I 2020 Predictive Maintenance Dataset to build machine learning models that can predict machine failures based on sensor data and operational parameters. The implementation includes comprehensive data analysis, feature engineering, and model evaluation.

## Dataset

The dataset contains 10,000 data points with the following features:
- UDI, Product ID
- Type (categorical: L, M, H)
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine failure (target variable)
- Failure modes: TWF, HDF, PWF, OSF, RNF

## Key Features

- Data loading and preprocessing
- Exploratory Data Analysis (EDA) with comprehensive visualizations
- Handling class imbalance using SMOTE
- Machine learning model training (Random Forest, XGBoost)
- Model evaluation (accuracy, classification report, confusion matrix)
- Feature importance visualization

## Project Structure

```
Predictive-Vehicle-Maintenance-Analyst/
├── data/
│   └── ai4i2020.csv
├── notebooks/
│   └── predictive_maintenance_analysis.ipynb
├── results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── xgboost_confusion_matrix.png
│   ├── xgboost_feature_importance.png
│   ├── xgboost_log_loss.png
│   └── xgboost_error.png
├── README.md
└── requirements.txt
```

## Requirements

To run this project, you'll need:
- Python 3. 
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Results

The analysis provides:
- Visualizations of data distribution and feature relationships
- Improved model performance after balancing classes with SMOTE
- Insights into important features for predicting machine failures
- Comparison of Random Forest and XGBoost classifier performance

## Usage

1. Clone the repository
2. Install the required packages
3. Run the Jupyter notebook `predictive_maintenance_analysis.ipynb`
4. View the results in the `results` folder

This project demonstrates practical application of machine learning for predictive maintenance in automotive and industrial contexts.

