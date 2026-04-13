from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("organ_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    input_data = pd.DataFrame([{
        "Donor_Age": int(data["donor_age"]),
        "Recipient_Age": int(data["recipient_age"]),
        "Donor_Blood": int(data["donor_blood"]),
        "Recipient_Blood": int(data["recipient_blood"]),
        "Organ_Type": int(data["organ_type"]),
        "Urgency_Level": int(data["urgency"]),
        "HLA_Match": int(data["hla"]),
        "Severity_Score": int(data["severity"]),
        "Donor_Organ_Quality": int(data["quality"]),
        "Compatibility_Score": int(data["score"])
    }])

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    result = "Suitable Match" if prediction[0] == 1 else "Not Suitable Match"
    confidence = round(max(prob[0]), 2)

    return render_template('index.html', prediction=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)