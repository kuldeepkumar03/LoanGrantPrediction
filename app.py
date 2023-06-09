import numpy as np
import pandas as pd
from flask import Flask, request, render_template 
import pickle
import os
import xgboost as xgb

app = Flask(__name__)

def load_model():
    model = pickle.load(open('pkl_xgb_model.pkl', 'rb'))
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    labels = ['Rejected', 'Approved']
    cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
           'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    cat_colst = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    ohe_col = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Gender_Female', 'Gender_Male',
            'Married_No', 'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Graduate',
            'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']

    features = [x for x in request.form.values()]
    data = pd.DataFrame([np.array(features)], columns=cols)
    data[num_cols] = data[num_cols].astype(int)

    values = pd.DataFrame(columns=ohe_col, dtype=int)
    values.loc[0] = [0] * len(values.columns)

    values.loc[0][num_cols] = data.loc[0][num_cols]
    values.loc[0][['Gender_Female', 'Gender_Male']] = [1, 0] if data['Gender'][0] == 'Female' else [0, 1]
    values.loc[0][['Married_No', 'Married_Yes']] = [1, 0] if data['Married'][0] == 'No' else [0, 1]
    if data['Dependents'][0] == '0':
        values.loc[0]['Dependents_0'] = 1
    elif data['Dependents'][0] == '1':
        values.loc[0]['Dependents_1'] = 1
    elif data['Dependents'][0] == '2':
        values.loc[0]['Dependents_2'] = 1
    elif data['Dependents'][0] == '3+':
        values.loc[0]['Dependents_3+'] = 1
    values.loc[0]['Education_Graduate'] = 1 if data['Education'][0] == 'Graduate' else 0
    values.loc[0]['Self_Employed_Yes'] = 1 if data['Self_Employed'][0] == 'Yes' else 0
    if data['Property_Area'][0] == 'Rural':
        values.loc[0]['Property_Area_Rural'] = 1
    elif data['Property_Area'][0] == 'Semiurban':
        values.loc[0]['Property_Area_Semiurban'] = 1
    elif data['Property_Area'][0] == 'Urban':
        values.loc[0]['Property_Area_Urban'] = 1

    model = load_model()
    prediction = model.predict(values)
    result = labels[prediction[0]]

    return render_template('index.html', output='Loan {}'.format(result))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)