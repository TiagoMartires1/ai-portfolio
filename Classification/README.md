# Churn Prediction Project

## Project Objective
Build a supervised machine learning model to predict customer churn and evaluate different classification approaches.

## Learning Goals
- Practice end-to-end supervised ML
- Compare baseline vs ensemble models
- Focus on evaluation metrics relevant to churn
- Write modular, reusable ML code

## Project Structure
- `notebooks/`: Exploratory data analysis and initial experimentation
- `src/`: Reproducible training, preprocessing, and evaluation scripts
- `data/`: Input datasets used in the project (included for reproducibility)

## Dataset
- 7043 customers
- 23 features

The dataset used in this project is the Telco Customer Churn dataset,
publicly available on Kaggle.

Source: Kaggle – Telco Customer Churn  
Author: Blastchar  
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset is included in this repository for educational
and portfolio purposes.

## Preprocessing
- Handled missing values
- Encoded categorical variables
- Scaled numerical variables

## Models
1. Logistic Regression
2. Random Forest (tuned and not tuned)
3. Gradient Boosting

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Train vs Test comparison
- Cross-validation
- Confusion matrices and ROC curves included

## Conclusions
- Logistic Regression: stable baseline
- Random Forest: best recall, slightly lower precision, good generalization
- Gradient Boosting: strong potential but needs careful tuning

## Next Steps / Improvements
- Hyperparameter search
- Feature engineering
- Threshold optimization