# Titanic Survival Prediction 🚢

Predicting whether a passenger survived the Titanic disaster using machine learning, based on features like age, gender, ticket class, and fare.

## Overview

This project walks through a complete data science pipeline: exploring the data, cleaning it, engineering features, training multiple models, and comparing their performance.

## Dataset

The classic [Titanic dataset](https://www.kaggle.com/c/titanic) containing passenger details such as age, sex, passenger class, fare, and survival status.

## Steps Followed

1. **Exploratory Data Analysis (EDA)** — visualized survival counts, gender distribution, passenger class distribution, and age distribution
2. **Missing Value Handling** — filled missing Age (median), Embarked (mode), and Fare (median) values; visualized missing data with a heatmap
3. **Data Cleaning** — dropped irrelevant columns (Name, Ticket, Cabin, PassengerId)
4. **Encoding** — converted categorical features (Sex, Embarked) into numeric form using Label Encoding
5. **Model Training** — trained and compared three models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
6. **Evaluation** — assessed models using accuracy score, confusion matrix, and classification report
7. **Feature Importance** — identified which features (e.g. Sex, Passenger Class) most influenced survival predictions
8. **Model Saving** — saved the best-performing model using `joblib` for future use

## Tools & Libraries

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib

## Results

Random Forest performed the best among the three models. See the notebook for detailed accuracy comparisons and visualizations.

## What I Learned

This project helped me understand the full data science workflow — from raw data to a working, evaluated model — including handling missing data, encoding categorical variables, and comparing model performance.
