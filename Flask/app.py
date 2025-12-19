import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 1. Load artifacts
print("Loading models...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders_dict = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. MAPPINGS
# Map HTML names -> Notebook names
KEY_MAPPING = {
    'education_num': 'education.num',
    'marital_status': 'marital.status',
    'hours_per_week': 'hours.per.week',
    'native_country': 'native.country'
}

# 3. FINAL COLUMN ORDER (14 Columns)
# This must match the original dataset structure exactly
EXPECTED_COLS = [
    'age', 
    'workclass', 
    'fnlwgt',          # Missing column 1
    'education', 
    'education.num', 
    'marital.status', 
    'occupation', 
    'relationship', 
    'race', 
    'sex', 
    'capital.gain',    # Missing column 2
    'capital.loss',    # Missing column 3
    'hours.per.week', 
    'native.country'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.form.to_dict()
        
        # Rename keys
        processed_data = {}
        for key, value in raw_data.items():
            new_key = KEY_MAPPING.get(key, key)
            processed_data[new_key] = value

        input_df = pd.DataFrame([processed_data])

        # Add missing columns
        input_df['fnlwgt'] = 180000 # Use mean weight instead of 0
        input_df['capital.gain'] = 0
        input_df['capital.loss'] = 0

        input_df = input_df[EXPECTED_COLS]

        # --- DEBUGGING / FIXING STRINGS ---
        for col in input_df.columns:
            if col in encoders_dict:
                le = encoders_dict[col]
                
                # CHECK 1: See if the model expects spaces (e.g., " Private")
                # We check the first class in the encoder to see if it starts with space
                model_expects_space = str(le.classes_[0]).startswith(' ')
                
                val = input_df.iloc[0][col]
                
                # If model wants space but we don't have it, add it
                if model_expects_space and not str(val).startswith(' '):
                    input_df.at[0, col] = " " + str(val)
                # If model does NOT want space but we have it, strip it
                elif not model_expects_space and str(val).startswith(' '):
                    input_df.at[0, col] = str(val).strip()

                # Print to terminal to see what is happening (DEBUGGING)
                print(f"Column: {col} | Value sent: '{input_df.iloc[0][col]}' | In Encoder? {input_df.iloc[0][col] in le.classes_}")

                # Encode
                input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                input_df[col] = le.transform(input_df[col])
            else:
                input_df[col] = pd.to_numeric(input_df[col])

        final_features = scaler.transform(input_df.values)
        
        # Print probabilities to see how close it is
        prediction_prob = model.predict_proba(final_features)
        print(f"Probabilities: {prediction_prob}") 

        prediction = model.predict(final_features)
        result_text = ">50K" if prediction[0] == 1 else "<=50K"

        return jsonify({'prediction_text': result_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)