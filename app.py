from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for local testing

# Load the trained model
try:
    model = joblib.load('optimized_xgb_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model file 'optimized_xgb_model.pkl' not found. Ensure it is in the same directory as app.py.")

# Define the expected features (same as X_train in your notebook)
expected_features = [
    'Age', 'ChronicDiseaseCount', 'LengthOfStay', 'EmergencyVisit', 'InpatientVisit',
    'OutpatientVisit', 'TotalVisits', 'BMIMin', 'BMIMax', 'BMIMedian', 'BMIMean',
    'BPDiastolicMin', 'BPDiastolicMax', 'BPDiastolicMedian', 'BPDiastolicMean',
    'BPSystolicMin', 'BPSystolicMax', 'BPSystolicMedian', 'BPSystolicMean',
    'TemperatureMin', 'TemperatureMax', 'TemperatureMedian', 'TemperatureMean',
    'HeartRateMin', 'HeartRateMax', 'HeartRateMedian', 'HeartRateMean',
    'PulseRateMin', 'PulseRateMax', 'PulseRateMedian', 'PulseRateMean',
    'RespiratoryRateMin', 'RespiratoryRateMax', 'RespiratoryRateMedian', 'RespiratoryRateMean',
    'ACEInhibitors', 'ARBs', 'BetaBlockers', 'Diuretics', 'TotalMedicine',
    'CardiacTroponin', 'Hemoglobin', 'SerumSodium', 'SerumCreatinine', 'BNP', 'NT-proBNP',
    'DischargeDisposision_Home', 'DischargeDisposision_Hospice - Home', 'DischargeDisposision_SNF',
    'Gender_Male',
    'Race_White', 'Race_Unknown',
    'DiabetesMellitus_DM', 'DiabetesMellitus_Unknown',
    'ChronicKidneyDisease_CKD', 'ChronicKidneyDisease_Unknown',
    'Anemia_Anemia', 'Anemia_Unknown',
    'Depression_Depression', 'Depression_Unknown',
    'ChronicObstructivePulmonaryDisease_COPD', 'ChronicObstructivePulmonaryDisease_Unknown'
]

def validate_blood_pressure(bp):
    try:
        systolic, diastolic = map(float, bp.split('/'))
        if systolic < 70 or systolic > 200:
            return False, "Systolic blood pressure must be between 70 and 200 mmHg."
        if diastolic < 40 or diastolic > 120:
            return False, "Diastolic blood pressure must be between 40 and 120 mmHg."
        if systolic <= diastolic:
            return False, "Systolic blood pressure must be greater than diastolic blood pressure."
        return True, (systolic, diastolic)
    except ValueError:
        return False, "Invalid blood pressure format. Use systolic/diastolic (e.g., 120/80)."

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    try:
        data = request.json
    except Exception as e:
        return jsonify({'error': 'Invalid JSON data in request.'}), 400

    # Extract form inputs with default values
    age = float(data.get('age', 0))
    joining_date = data.get('joiningDate', '')
    condition = data.get('condition', '')
    blood_pressure = data.get('bloodPressure', '120/80')
    chronic_disease_count = float(data.get('chronicDiseaseCount', 0))
    length_of_stay = float(data.get('lengthOfStay', 0))
    emergency_visit = float(data.get('emergencyVisit', 0))
    inpatient_visit = float(data.get('inpatientVisit', 0))
    outpatient_visit = float(data.get('outpatientVisit', 0))
    bmi = float(data.get('bmi', 0))
    heart_rate = float(data.get('heartRate', 0))
    respiratory_rate = float(data.get('respiratoryRate', 0))
    temperature = float(data.get('temperature', 0))
    hemoglobin = float(data.get('hemoglobin', 0))
    serum_sodium = float(data.get('serumSodium', 0))
    serum_creatinine = float(data.get('serumCreatinine', 0))
    bnp = float(data.get('bnp', 0))
    nt_probnp = float(data.get('ntProbnp', 0))
    cardiac_troponin = float(data.get('cardiacTroponin', 0))
    ace_inhibitors = float(data.get('aceInhibitors', 0))
    arbs = float(data.get('arbs', 0))
    beta_blockers = float(data.get('betaBlockers', 0))
    diuretics = float(data.get('diuretics', 0))
    total_medicine = float(data.get('totalMedicine', 0))
    discharge_disposition = data.get('dischargeDisposition', 'Home')
    gender = data.get('gender', 'Male')
    race = data.get('race', 'White')
    diabetes_mellitus = data.get('diabetesMellitus', 'DM')
    chronic_kidney_disease = data.get('chronicKidneyDisease', 'CKD')
    anemia = data.get('anemia', 'Anemia')
    depression = data.get('depression', 'Depression')
    copd = data.get('copd', 'COPD')

    # Validate blood pressure
    bp_valid, bp_result = validate_blood_pressure(blood_pressure)
    if not bp_valid:
        return jsonify({'error': bp_result}), 400
    systolic, diastolic = bp_result

    # Calculate days since joining
    try:
        today = datetime.strptime('2025-03-21', '%Y-%m-%d')
        join_date = datetime.strptime(joining_date, '%Y-%m-%d')
        days_since_joining = max(0, (today - join_date).days)
        if days_since_joining > 365:
            days_since_joining = 365  # Cap at 365 days to avoid unrealistic values
    except ValueError:
        return jsonify({'error': 'Invalid joining date format. Use YYYY-MM-DD.'}), 400

    # Validate condition
    valid_conditions = ['heart', 'diabetes', 'lung', 'other']
    if condition not in valid_conditions:
        return jsonify({'error': f"Invalid condition. Must be one of {valid_conditions}."}), 400

    # Prepare numerical inputs
    input_data = {
        'Age': age,
        'ChronicDiseaseCount': chronic_disease_count,
        'LengthOfStay': length_of_stay,
        'EmergencyVisit': emergency_visit,
        'InpatientVisit': inpatient_visit,
        'OutpatientVisit': outpatient_visit,
        'TotalVisits': emergency_visit + inpatient_visit + outpatient_visit,
        'BMIMin': bmi,
        'BMIMax': bmi,
        'BMIMedian': bmi,
        'BMIMean': bmi,
        'BPDiastolicMin': diastolic,
        'BPDiastolicMax': diastolic,
        'BPDiastolicMedian': diastolic,
        'BPDiastolicMean': diastolic,
        'BPSystolicMin': systolic,
        'BPSystolicMax': systolic,
        'BPSystolicMedian': systolic,
        'BPSystolicMean': systolic,
        'TemperatureMin': temperature,
        'TemperatureMax': temperature,
        'TemperatureMedian': temperature,
        'TemperatureMean': temperature,
        'HeartRateMin': heart_rate,
        'HeartRateMax': heart_rate,
        'HeartRateMedian': heart_rate,
        'HeartRateMean': heart_rate,
        'PulseRateMin': heart_rate,  # Assuming pulse rate = heart rate
        'PulseRateMax': heart_rate,
        'PulseRateMedian': heart_rate,
        'PulseRateMean': heart_rate,
        'RespiratoryRateMin': respiratory_rate,
        'RespiratoryRateMax': respiratory_rate,
        'RespiratoryRateMedian': respiratory_rate,
        'RespiratoryRateMean': respiratory_rate,
        'ACEInhibitors': ace_inhibitors,
        'ARBs': arbs,
        'BetaBlockers': beta_blockers,
        'Diuretics': diuretics,
        'TotalMedicine': total_medicine,
        'CardiacTroponin': cardiac_troponin,
        'Hemoglobin': hemoglobin,
        'SerumSodium': serum_sodium,
        'SerumCreatinine': serum_creatinine,
        'BNP': bnp,
        'NT-proBNP': nt_probnp,
    }

    # Prepare categorical inputs
    categorical_data = {
        'DischargeDisposision': discharge_disposition,
        'Gender': gender,
        'Race': race,
        'DiabetesMellitus': diabetes_mellitus,
        'ChronicKidneyDisease': chronic_kidney_disease,
        'Anemia': anemia,
        'Depression': depression,
        'ChronicObstructivePulmonaryDisease': copd
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    categorical_df = pd.DataFrame([categorical_data])
    categorical_df = pd.get_dummies(categorical_df, drop_first=True)

    # Combine numerical and categorical data
    input_df = pd.concat([input_df, categorical_df], axis=1)

    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[expected_features]

    # Make prediction
    try:
        probability = model.predict_proba(input_df)[:, 1][0]
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Categorize risk
    if probability <= 0.3:
        risk_level = 'Low'
    elif probability <= 0.7:
        risk_level = 'Moderate'
    else:
        risk_level = 'High'

    # Prepare feature importance (simplified)
    features = {
        'Age': 'High risk' if age > 70 else 'Moderate risk' if age > 60 else 'Low risk',
        'DaysSinceJoining': 'High risk' if days_since_joining < 10 else 'Moderate risk' if days_since_joining < 30 else 'Low risk',
        'Condition': 'High risk' if condition in ['heart', 'diabetes'] else 'Moderate risk' if condition == 'lung' else 'Low risk',
        'BloodPressure': 'High risk' if systolic > 140 or diastolic > 90 else 'Low risk'
    }

    return jsonify({
        'probability': float(probability),
        'riskLevel': risk_level,
        'features': features,
        'daysSinceJoining': days_since_joining
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)