<div align="center">

# Loan Eligibility Prediction (Logistic Regression)

Predict loan approval using a trained Logistic Regression model, a Flask web UI, and optional AI-generated explanations (Gemini). The project follows CRISP-DM and includes training notebooks and a ready-to-run web app.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-black)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

## Overview

This repository builds a binary classifier to predict whether a loan application is likely to be approved. It includes:
- A Jupyter notebook for data prep, model training, evaluation, and model export.
- A Flask application that serves a web interface where users can submit details and get a prediction.
- An optional AI explanation flow using Google Gemini that summarizes the decision in plain language.

Dataset: https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset

## Repository Structure

```
├─ Model/
│  ├─ Loan-Eligiblility-Prediction.ipynb     # training & evaluation
│  ├─ loan-train.csv                         # training data
│  ├─ loan-test.csv                          # test data (no labels)
│  └─ loan_eligibility_model.pkl             # exported pipeline/model
├─ model_deployment/
│  ├─ app.py                                 # Flask app
│  └─ interpret_w_and_ratio.py               # weights/odds utilities (optional)
├─ interface/
│  ├─ index.html                             # input form
│  ├─ result.html                            # prediction + AI summary
│  └─ history.html                           # query past predictions
├─ token.py                                   # (optional) custom token utils
├─ README.md
└─ LICENSE
```

## Features

- Logistic Regression classifier (with optional preprocessing pipeline)
- Clean, modern web UI (Predict, Result, History)
- Optional AI explanation via Google Gemini
- MySQL persistence of predictions and summaries

## Prerequisites

- Python 3.10+ (Anaconda or python.org)
- MySQL Server (local or remote)
- A Google AI Studio API key (if you want AI summaries)

## Quickstart (Windows PowerShell)

1) Clone this repo and enter the folder
```powershell
git clone https://github.com/<your-username>/Loan-Eligibility-Prediction-Using-Logistics-Regression-Algorithm.git
cd Loan-Eligibility-Prediction-Using-Logistics-Regression-Algorithm
```

2) Create a virtual environment and activate it (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies
```powershell
pip install --upgrade pip
pip install flask python-dotenv google-generativeai joblib numpy pandas scikit-learn mysql-connector-python
```

4) Set environment variables (.env at repo root)
```
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=replace_with_random_secret
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=loan_prediction_db
```

5) Create the database and table
```sql
CREATE DATABASE IF NOT EXISTS loan_prediction_db;
USE loan_prediction_db;

CREATE TABLE IF NOT EXISTS predictions (
	id INT AUTO_INCREMENT PRIMARY KEY,
	user_name VARCHAR(100) NOT NULL,
	gender VARCHAR(10),
	married VARCHAR(10),
	dependents VARCHAR(10),
	education VARCHAR(30),
	self_employed VARCHAR(10),
	applicant_income DECIMAL(12,2),
	coapplicant_income DECIMAL(12,2),
	loan_amount DECIMAL(12,2),
	loan_amount_term INT,
	credit_history TINYINT,
	property_area VARCHAR(30),
	prediction VARCHAR(20),
	weights_json JSON NULL,
	odds_ratios_json JSON NULL,
	ai_summary TEXT NULL,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

6) Ensure the model file exists

- By default, the app loads `Model/loan_eligibility_model.pkl`. If you haven’t exported it yet, open the notebook and save the trained pipeline with joblib.

## Run the App

From the repo root:
```powershell
python model_deployment\app.py
```

Open http://127.0.0.1:5000 in your browser.

## Training (Notebook)

- Open `Model/Loan-Eligiblility-Prediction.ipynb`.
- Explore data, handle missing values, encode features (or use a ColumnTransformer), split data, train Logistic Regression.
- Evaluate with accuracy, precision, recall, AUC; visualize confusion matrix and ROC.
- Export the trained model:
```python
import joblib
joblib.dump(pipe, 'loan_eligibility_model.pkl')
```
Move the model file into `Model/` if needed.

## API / Routes

- GET `/` — Show the prediction form
- POST `/predict` — Run prediction and (optionally) generate AI explanation; save to DB
- GET `/history` — Search predictions by user name

## Configuration Notes

- The app loads templates from `interface/` via Flask’s `template_folder` config.
- The model path is resolved relative to repo root: `Model/loan_eligibility_model.pkl`.
- If your pipeline includes a ColumnTransformer with OneHotEncoder, inputs can be raw strings; otherwise the app normalizes values to match training encodings.
- Gemini model id used: `gemini-1.5-flash` (configure with `GEMINI_API_KEY`).

## Troubleshooting

- 500 error during predict with OneHotEncoder
	- Ensure your scikit-learn version matches the one used during training; try pinning `scikit-learn==1.3.2` and `numpy<2.0`.
	- Make sure the model was saved with consistent preprocessing and that inputs are normalized (Dependents as "0/1/2/3+", Property_Area as "Rural/Semiurban/Urban").

- Module import errors
	- Install missing packages: `pip install flask python-dotenv google-generativeai joblib numpy pandas scikit-learn mysql-connector-python`

- Templates not found
	- The app sets `template_folder` to `interface/`. Use `render_template('index.html')`, etc.

- Database errors
	- Check `.env` DB_* variables and that the `predictions` table exists.
	- If you don’t need history, temporarily disable DB insert/select blocks.

## Security

- Do NOT commit `.env` or secrets. Rotate keys if they were exposed.
- Consider enabling SSL for remote DB connections.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- Kaggle Dataset by Vikas Ukani.
- scikit-learn, Flask, and the Python community.