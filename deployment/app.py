
from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the tuned 8-feature Decision Tree
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "decision_tree_tuned_8features.pkl"
)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Must match the exact feature order used during training
FEATURES = [
    "education",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value"
]

# Match the encoding used during training
education_mapping = {
    "Graduate": 0,
    "Not Graduate": 1
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            education = request.form["education"]

            input_data = {
                "education": education_mapping[education],
                "income_annum": float(request.form["income_annum"]),
                "loan_amount": float(request.form["loan_amount"]),
                "loan_term": float(request.form["loan_term"]),
                "cibil_score": float(request.form["cibil_score"]),
                "residential_assets_value": float(
                    request.form["residential_assets_value"]
                ),
                "commercial_assets_value": float(
                    request.form["commercial_assets_value"]
                ),
                "luxury_assets_value": float(
                    request.form["luxury_assets_value"]
                )
            }

            # Preserve the exact training feature order
            new_data = pd.DataFrame(
                [input_data],
                columns=FEATURES
            )

            # Predict using the tuned model
            result = int(model.predict(new_data)[0])

            if result == 0:
                prediction = "Loan Approved"
            elif result == 1:
                prediction = "Loan Rejected"
            else:
                error = "Unexpected prediction returned by the model."

        except (ValueError, KeyError):
            error = "Please enter valid values in all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
