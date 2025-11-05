import json
import numpy as np
import joblib

# load the model
pipe_model = joblib.load("loan_eligibility_model.pkl")

# Extract the logistic regression part
logreg = pipe_model.named_steps['logisticregression']

# Define feature names (must match training order!)
feature_names = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Get coefficients (weights)
coefficients = logreg.coef_[0]

# Compute odds ratios
odd_ratios = np.exp(coefficients)

# create dictionaries for storage (as JSON)
weights_dict = dict(zip(feature_names, coefficients))
odds_dict = dict(zip(feature_names, odd_ratios))

# Convert to JSON strings for DB storage
weights_json = json.dumps(weights_dict)
odds_json = json.dumps(odds_dict)
"""
- Converts python dict to JSON text
- Useful for storing in a database or sending to frontend.

"""

# Example interpretation string (for loggin or basico output)
interpretation = []
for feature, coef, odds in zip(feature_names, coefficients, odd_ratios):
    direction = "increase" if coef > 0 else "decrease"
    interpretation.append(f"{feature}: Coefficient {coef: .4f} ({direction} log-odds), Odds Ratio {odds: .4f}")

print("\n".join(interpretation))

"""
Sample printed output:

Credit_History: Coefficient 2.1234 (increases log-odds), Odds Ratio 8.3612
LoanAmount: Coefficient -0.5032 (decreases log-odds), Odds Ratio 0.6045

"""