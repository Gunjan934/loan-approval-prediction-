
from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the saved Decision Tree model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "decision_tree_model.pkl"
)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# These mappings must match your notebook's encoding
education_mapping = {
    "Graduate": 0,
    "Not Graduate": 1
}

self_employed_mapping = {
    "No": 0,
    "Yes": 1
}

# Test-set metrics from your notebook
model_metrics = {
    "accuracy": 97.66,
    "precision": 94.96,
    "recall": 99.07,
    "f1_score": 96.97
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            input_data = {
                "no_of_dependents": int(
                    request.form["no_of_dependents"]
                ),
                "education": education_mapping[
                    request.form["education"]
                ],
                "self_employed": self_employed_mapping[
                    request.form["self_employed"]
                ],
                "income_annum": float(
                    request.form["income_annum"]
                ),
                "loan_amount": float(
                    request.form["loan_amount"]
                ),
                "loan_term": int(
                    request.form["loan_term"]
                ),
                "cibil_score": float(
                    request.form["cibil_score"]
                ),
                "residential_assets_value": float(
                    request.form["residential_assets_value"]
                ),
                "commercial_assets_value": float(
                    request.form["commercial_assets_value"]
                ),
                "luxury_assets_value": float(
                    request.form["luxury_assets_value"]
                ),
                "bank_asset_value": float(
                    request.form["bank_asset_value"]
                )
            }

            if any(
                value < 0
                for key, value in input_data.items()
                if key not in ("education", "self_employed")
            ):
                raise ValueError(
                    "Numeric values cannot be negative."
                )

            if input_data["loan_term"] == 0:
                raise ValueError(
                    "Loan term must be greater than zero."
                )

            if input_data["cibil_score"] > 900:
                raise ValueError(
                    "CIBIL score must not exceed 900."
                )

            # Keep the original feature order
            feature_columns = [
                "no_of_dependents",
                "education",
                "self_employed",
                "income_annum",
                "loan_amount",
                "loan_term",
                "cibil_score",
                "residential_assets_value",
                "commercial_assets_value",
                "luxury_assets_value",
                "bank_asset_value"
            ]

            new_data = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            result = int(model.predict(new_data)[0])

            if result == 0:
                prediction = {
                    "label": "Loan Approved",
                    "status": "approved",
                    "description": (
                        "The model predicts that this "
                        "application belongs to the "
                        "approved class."
                    )
                }
            elif result == 1:
                prediction = {
                    "label": "Loan Rejected",
                    "status": "rejected",
                    "description": (
                        "The model predicts that this "
                        "application belongs to the "
                        "rejected class."
                    )
                }
            else:
                raise ValueError(
                    "Unexpected prediction label."
                )

        except (ValueError, KeyError) as exc:
            error = str(exc) or "Please check your input values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        metrics=model_metrics
    )


if __name__ == "__main__":
    app.run(debug=True)
