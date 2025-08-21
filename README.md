# Predictive-Vehicle-Maintenance-Analyst
A data science project analyzing sensor data to predict vehicle component failure. showcases the full ML pipeline: data cleaning, exploratory analysis, feature engineering, and model training with a Random Forest and xboost classifier. Built with Python, Pandas, and Scikit-learn


This Jupyter Notebook performs predictive maintenance analysis on the "AI4I 2020 Predictive Maintenance Dataset". The goal is to predict machine failures based on sensor data and operational parameters.

## Dataset
The dataset contains 10,000 rows and 14 columns, including:
- UDI, Product ID
- Type (categorical: L, M, H)
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine failure (target variable)
- Failure modes: TWF, HDF, PWF, OSF, RNF

## Features
Data loading and preprocessing
Exploratory Data Analysis (EDA) with visualizations
Handling class imbalance using SMOTE
Machine learning model training (Random Forest, XGBoost)
Model evaluation (accuracy, classification report, confusion matrix)
Feature importance visualization


## Files

predictive_maintenance/
├── ai4i2020.csv
├── predictive_maintainance_analysis.ipynb
├── requirements.txt
├── README.md
└── images/
    ├── failure_distribution.png
    ├── tool_wear_vs_failure.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    └── feature_importance.png

    
## Requirements
Python 3.7+
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost



## Results
-Visualizations of data distribution and feature relationships
- a more Improved model performance after balancing classes with SMOTE
-Insights into important features for predicting machine failure


