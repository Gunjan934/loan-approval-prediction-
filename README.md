# Loan Approval Prediction

A Machine Learning classification project that predicts whether a loan application is likely to be approved based on the applicant's financial, employment, credit, and asset-related information.

## 📌 About the Project

Loan approval depends on several factors such as annual income, loan amount, loan term, credit score, employment status, education, and financial assets.

The goal of this project is to analyze a loan approval dataset, understand the factors associated with loan approval, preprocess the data, and build classification models to predict the loan approval status.

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Understand the dataset and its features
- Handle and prepare categorical data
- Identify potential outliers and relationships between features
- Prepare data for Machine Learning
- Train multiple classification models
- Evaluate model performance using different metrics
- Compare the models and identify the better-performing approach
- Check the model for potential overfitting

## 📊 Dataset

The dataset contains information about loan applicants and their financial background.

### Features

| Feature | Description |
|---|---|
| `loan\_id` | Unique identifier of the loan application |
| `no\_of\_dependents` | Number of dependents of the applicant |
| `education` | Education level of the applicant |
| `self\_employed` | Employment status of the applicant |
| `income\_annum` | Annual income of the applicant |
| `loan\_amount` | Requested loan amount |
| `loan\_term` | Loan repayment term |
| `cibil\_score` | Credit score of the applicant |
| `residential\_assets\_value` | Value of residential assets |
| `commercial\_assets\_value` | Value of commercial assets |
| `luxury\_assets\_value` | Value of luxury assets |
| `bank\_asset\_value` | Value of bank assets |
| `loan\_status` | Target variable representing loan approval status |

## 🔍 Exploratory Data Analysis

The dataset was analyzed to understand its structure and identify patterns that may influence loan approval.

The EDA includes:

- Dataset dimensions and structure
- Data types and statistical information
- Missing value analysis
- Duplicate value checking
- Unique value analysis
- Target variable distribution
- Numerical feature distributions
- Categorical feature analysis
- Correlation analysis
- Outlier detection using boxplots

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Cleaned column names by removing unnecessary whitespace
- Removed `loan\_id` because it is only an identifier and does not contribute useful predictive information
- Separated input features and the target variable
- Encoded categorical variables
- Split the dataset into training and testing sets
- Applied feature scaling using `StandardScaler`

## 🤖 Machine Learning Models

Three classification algorithms were implemented and compared:

### Logistic Regression

Used as a baseline classification algorithm to predict whether a loan application will be approved.

### K-Nearest Neighbors (KNN)

Predicts the loan status based on the similarity between a new application and existing applications in the dataset.

### Gaussian Naive Bayes

Uses probability-based classification to predict the likelihood of a loan application belonging to the approved or rejected class.

## 📈 Model Evaluation

The models are evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The performance of the models is compared to determine which algorithm provides the most suitable results for the dataset.

## 🔎 Overfitting Analysis

Training and testing performance are compared to check whether the models generalize well to unseen data.

The project also considers the difference between training and testing accuracy when evaluating potential overfitting.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📁 Project Structure

```text

loan-approval-prediction/

│

├── .gitignore

├── Loan approval prediction.ipynb

├── loan\_approval\_dataset.csv

└── README.md