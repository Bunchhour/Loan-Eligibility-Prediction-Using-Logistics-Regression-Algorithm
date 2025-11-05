import os
import joblib
import json
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
from flask import Flask, render_template, request, redirect, url_for, flash

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Add it to .env or your environment variables.")

# set up api key
genai.configure(api_key=api_key)

# create model instance (use a valid Gemini model id)
model = genai.GenerativeModel('gemini-2.5-flash')

# Resolve project base directory (repo root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Load the model robustly from the Model folder
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'loan_eligibility_model.pkl')
model_pipe = joblib.load(MODEL_PATH)

# Feature names (must match training order)
feature_names = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

def generate_loan_explanation(prediction, user_inputs_dict, weights_dict, odds_dict):
    """
    Generate AI explanation for loan decision
    
    Args:
        prediction: str - "Approved" or "Denied"
        user_inputs_dict: dict - customer's input data
        weights_dict: dict - model coefficients
        odds_dict: dict - odds ratios
    
    Returns:
        str - AI-generated explanation
    """
    customer_inputs_str = json.dumps(user_inputs_dict, indent=2)
    weights_str = json.dumps(weights_dict, indent=2)
    odds_str = json.dumps(odds_dict, indent=2)

    prompt = f"""
You are an AI financial assistant that explains loan approval decisions clearly and fairly.

### Loan Decision Summary Task
A logistic regression model has evaluated a customer's loan application.

Your goal:
- Explain why the loan was **{prediction.lower()}**
- Use the customer's input data, model coefficients, and odds ratios
- Highlight the most influential factors (both positive and negative)
- Be friendly, neutral, and supportive — no technical jargon

### Model Output

**Prediction:** {prediction}

**Customer Inputs:**
{user_inputs_dict}

**Model Weights (Coefficients):**
{weights_dict}

**Model Odds Ratios:**
{odds_dict}

### How to Explain
- If odds ratio > 1 → this increased approval likelihood  
- If odds ratio < 1 → this decreased approval likelihood  
- Mention ~3–5 key factors only  
- Avoid listing all values — summarize impact
- If denied, provide encouraging guidance on how to improve chances next time  

### Format
Provide:
1) One-paragraph explanation in simple language
2) Bullet-point key factors
3) Supportive actionable tip(s) for the customer

### Tone examples
✅ Helpful, calm, encouraging  
✅ Easy for non-technical customers  
❌ No math, coefficients, or complex economic terms  

Now explain the loan result.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating explanation: {str(e)}"  # Fallback for production

# Example usage:
# explanation = generate_loan_explanation("Approved", {...}, {...}, {...})

#============================= Part 2: Dealing with Deployment ==================================
# Configure Flask to use the 'interface' folder as templates directory
TEMPLATE_DIR = os.path.join(BASE_DIR, 'interface')
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = os.getenv("SECRET_KEY")

# Database part
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# DB connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except Error as e:
        print(f"DB Error: {e}")
        return None


#============ app ===========================================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    user_name = request.form['user_name']
    user_inputs = {
        'Gender': request.form['gender'],
        'Married': request.form['married'],
        'Dependents': int(request.form['dependents']),
        'Education': request.form['education'],
        'Self_Employed': request.form['self_employed'],
        'ApplicantIncome': float(request.form['applicant_income']),
        'CoapplicantIncome': float(request.form['coapplicant_income']),
        'LoanAmount': float(request.form['loan_amount']),
        'Loan_Amount_Term': int(request.form['loan_amount_term']),
        'Credit_History': int(request.form['credit_history']),
        'Property_Area': request.form['property_area']
    }
    # Convert to DataFrame for prediction
    user_df = pd.DataFrame([user_inputs])
    # Ensure expected column order
    user_df = user_df[feature_names]

    # Detect if pipeline includes its own preprocessor (e.g., ColumnTransformer)
    def _has_preprocessor(pipe):
        try:
            return hasattr(pipe, 'named_steps') and 'preprocessor' in pipe.named_steps
        except Exception:
            return False

    # Encode categorical features to match LabelEncoder training used in the notebook
    def _encode_for_numeric_pipeline(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Normalize helper
        def norm(s):
            return str(s).strip().lower().replace('-', '').replace(' ', '')

        gender_map = {'female': 0, 'male': 1}
        married_map = {'no': 0, 'yes': 1}
        education_map = {'graduate': 0, 'notgraduate': 1}
        # Property area in dataset often: Rural, Semiurban, Urban
        property_map = {'rural': 0, 'semiurban': 1, 'urban': 2}
        self_emp_map = {'no': 0, 'yes': 1}

        d.loc[:, 'Gender'] = gender_map.get(norm(d['Gender'].iloc[0]), 0)
        d.loc[:, 'Married'] = married_map.get(norm(d['Married'].iloc[0]), 0)
        # Dependents mapping: 0,1,2,3+
        dep_raw = str(df['Dependents'].iloc[0])
        if dep_raw.endswith('+'):
            dep_code = 3
        else:
            try:
                dep_code = int(float(dep_raw))
                dep_code = 3 if dep_code >= 3 else dep_code
            except Exception:
                dep_code = 0
        d.loc[:, 'Dependents'] = dep_code
        d.loc[:, 'Education'] = education_map.get(norm(d['Education'].iloc[0]), 0)
        d.loc[:, 'Self_Employed'] = self_emp_map.get(norm(d['Self_Employed'].iloc[0]), 0)
        d.loc[:, 'Property_Area'] = property_map.get(norm(d['Property_Area'].iloc[0]), 0)
        # numeric columns already numeric
        return d

    # If model has an internal preprocessor (OneHotEncoder, etc.), normalize raw strings
    def _normalize_for_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # Normalize spaces/hyphens and capitalization where needed
        def norm_token(v: str) -> str:
            return str(v).strip()

        # Dependents in training were strings like '0','1','2','3+'
        dep = d['Dependents'].iloc[0]
        try:
            dep_int = int(float(dep))
            dep_str = '3+' if dep_int >= 3 else str(dep_int)
        except Exception:
            dep_s = str(dep)
            dep_str = '3+' if dep_s.startswith('3') else dep_s
        d.loc[:, 'Dependents'] = dep_str

        # Property_Area often 'Rural', 'Semiurban', 'Urban' in dataset
        prop_raw = str(d['Property_Area'].iloc[0]).strip().lower().replace('-', ' ').replace('_', ' ')
        if 'semi' in prop_raw:
            prop_norm = 'Semiurban'
        elif 'urban' in prop_raw and 'semi' not in prop_raw:
            prop_norm = 'Urban'
        elif 'rural' in prop_raw:
            prop_norm = 'Rural'
        else:
            prop_norm = str(d['Property_Area'].iloc[0])
        d.loc[:, 'Property_Area'] = prop_norm

        # Keep other categoricals as standard title case to match typical training values
        d.loc[:, 'Gender'] = str(d['Gender'].iloc[0]).strip().title()
        d.loc[:, 'Married'] = str(d['Married'].iloc[0]).strip().title()
        d.loc[:, 'Education'] = str(d['Education'].iloc[0]).strip().title()
        d.loc[:, 'Self_Employed'] = 'Yes' if str(d['Self_Employed'].iloc[0]).strip().lower() in ['yes', 'y', '1', 'true'] else 'No'
        return d

    if _has_preprocessor(model_pipe):
        df_for_pred = _normalize_for_preprocessor(user_df)
    else:
        df_for_pred = _encode_for_numeric_pipeline(user_df)

    # Predict
    proba = model_pipe.predict_proba(df_for_pred)[0]
    label = int(model_pipe.predict(df_for_pred)[0])
    prediction = "Approved" if label == 1 else 'Denied'

    # Extract weights and odds (works only for non-OHE pipeline with same feature count)
    weights_dict = {}
    odds_dict = {}
    try:
        logreg = model_pipe.named_steps['logisticregression']
        coefficients = logreg.coef_[0]
        odds_ratios = np.exp(coefficients)
        if len(coefficients) == len(feature_names):
            weights_dict = {k: float(v) for k, v in zip(feature_names, coefficients)}
            odds_dict = {k: float(v) for k, v in zip(feature_names, odds_ratios)}
    except Exception:
        pass

    # Get AI summary
    ai_summary = generate_loan_explanation(prediction, user_inputs, weights_dict, odds_dict)
    
    # Store in DB
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO predictions 
        (user_name, gender, married, dependents, education, self_employed, 
         applicant_income, coapplicant_income, loan_amount, loan_amount_term, 
         credit_history, property_area, prediction, weights_json, odds_ratios_json, ai_summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
    
        values = (
            user_name, user_inputs['Gender'], user_inputs['Married'], user_inputs['Dependents'],
            user_inputs['Education'], user_inputs['Self_Employed'], user_inputs['ApplicantIncome'],
            user_inputs['CoapplicantIncome'], user_inputs['LoanAmount'], user_inputs['Loan_Amount_Term'],
            user_inputs['Credit_History'], user_inputs['Property_Area'], prediction,
            json.dumps(weights_dict), json.dumps(odds_dict), ai_summary
        )
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        flash('Prediction saved to history!')

    # Return results
    return render_template('result.html', prediction=prediction, ai_summary=ai_summary)

@app.route('/history', methods=['GET','POST'])
def history():
    if request.method == 'POST':
        user_name = request.form['user_name']
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM predictions WHERE user_name = %s ORDER BY created_at DESC"
            cursor.execute(query, (user_name,))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return render_template('history.html', results=results, user_name=user_name)
    return render_template('history.html')

if __name__ == '__main__':
    app.run(debug=True)





