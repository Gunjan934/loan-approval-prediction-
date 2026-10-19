# Loan Approval Prediction System

A machine learning project that predicts loan approval status based on an applicant's financial information, credit score, education, employment status, and asset values. This project covers data preprocessing, supervised learning, model evaluation, and deployment using Flask.

## Project Overview

The Loan Approval Prediction System uses historical loan application data to predict whether an application belongs to the approved or rejected class.

The project explores multiple supervised machine learning algorithms, evaluates their performance, and deploys a trained Decision Tree model through a web application.

### Objectives

- Explore and understand the loan approval dataset.
- Perform data preprocessing and categorical encoding.
- Train and compare multiple supervised machine learning models.
- Evaluate models using classification metrics and cross-validation.
- Save the trained model using Python pickle.
- Build a Flask web interface for making predictions.

## Features

- Loan application form with financial and personal input fields.
- Machine learning prediction using a trained Decision Tree classifier.
- Model persistence using a `.pkl` file.
- Flask backend for processing user input and generating predictions.
- Responsive web interface built with HTML and CSS.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset

The project uses `loan_approval_dataset.csv`.

The dataset contains the following features:

| Feature | Description |
|---|---|
| `loan_id` | Unique application identifier; excluded from model features |
| `no_of_dependents` | Number of dependents |
| `education` | Applicant's education level |
| `self_employed` | Whether the applicant is self-employed |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan repayment term |
| `cibil_score` | Applicant's credit score |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | Target variable: approved or rejected |

## Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Pickle
- **Backend:** Flask
- **Frontend:** HTML, CSS
- **Development Environment:** Jupyter Notebook, Visual Studio Code
- **Version Control:** Git and GitHub

## Machine Learning Workflow

1. **Data exploration:** Inspect the dataset, features, target variable, and data distributions.
2. **Preprocessing:** Separate features and target, exclude the application ID, and encode categorical variables.
3. **Train-test split:** Divide the dataset into training and testing sets.
4. **Model training:** Train and evaluate multiple supervised learning algorithms.
5. **Model evaluation:** Compare accuracy, precision, recall, F1-score, confusion matrix, and cross-validation results.
6. **Model selection:** Select the Decision Tree model based on the evaluation performed.
7. **Model persistence:** Save the trained model using Python's pickle module.
8. **Deployment:** Load the saved model in Flask and use applicant inputs to generate predictions.

## Models Explored

The project explored multiple supervised learning algorithms, including:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree Classifier

The Decision Tree model was selected for the current Flask deployment based on the model evaluation conducted during development.

## Decision Tree Evaluation Results

The following results were obtained during testing of the Decision Tree model:

| Metric | Test Result |
|---|---:|
| Accuracy | 97.66% |
| Precision | 94.96% |
| Recall | 99.07% |
| F1-score | 96.97% |
| Mean 5-fold cross-validation accuracy | 96.16% |
| Mean 5-fold cross-validation precision | 94.72% |
| Mean 5-fold cross-validation recall | 95.50% |

**Note:** Precision, recall, and F1-score above are for the positive class used by scikit-learn's default binary classification metrics (encoded label `1`). Verify the target encoding to determine whether this represents Approved or Rejected. Cross-validation scores are calculated on the training data, while test metrics are calculated on the held-out test set.

These metrics describe performance on the evaluated dataset and do not guarantee correct predictions for every new application.

## Project Structure

```text
supervised-learning/
│
├── loan_approval_dataset.csv
├── decision_tree_model.pkl
├── Loan approval prediction.ipynb
├── README.md
│
└── deployment/
    ├── app.py
    ├── venv/
    ├── templates/
    │   └── index.html
    └── static/
        └── style.css
```

The saved model is currently stored in the project root, one level above the `deployment` folder. The Flask application loads it using a path relative to `app.py`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "supervised learning"
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

On Windows:

```bash
python -m venv deployment/venv
```

### 3. Activate the environment

```powershell
.\deployment\venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install flask pandas scikit-learn
```

For reproducible results, use a compatible scikit-learn version with the one used to train the saved model.

### 5. Run the Flask application

```bash
cd deployment
python app.py
```

Open the local address shown in the terminal, usually:

`http://127.0.0.1:5000/`

Enter the applicant's details in the form and submit them to view the model's prediction.

## Model Persistence Using Pickle

The trained Decision Tree model is saved as `decision_tree_model.pkl`.

```python
import pickle

with open("decision_tree_model.pkl", "wb") as file:
    pickle.dump(model_DT, file)
```

The Flask application loads the saved model rather than training it every time the website starts.

The saved model file does not automatically include separately fitted preprocessing objects. The categorical encoding and feature order used in Flask must match those used during model training.

**Security note:** Only load pickle files from trusted sources.

## Future Improvements

- Add a model comparison dashboard.
- Display evaluation metrics and confusion matrices in the web interface.
- Improve input validation and user feedback.
- Save prediction history in a database.
- Add automated tests for preprocessing and predictions.
- Deploy the application to a hosting platform.

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions are generated from historical data and should not be treated as actual financial advice or a lender's final approval decision.

## Author

**Gunjan**

B.Tech Computer Science and Engineering

Interested in Python development, data analysis, and machine learning.

---

If you find this project useful, feel free to explore the code and suggest improvements.