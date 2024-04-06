import pandas as pd
from flask import Flask, request, render_template
import joblib

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model_new_classifier.pkl')

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input values from the HTML form
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        isFlaggedFraud = float(request.form['isFlaggedFraud'])
        CASH_OUT = float(request.form['CASH_OUT'])
        DEBIT = float(request.form['DEBIT'])
        NOTYPE = float(request.form['NOTYPE'])
        PAYMENT = float(request.form['PAYMENT'])
        TRANSFER = float(request.form['TRANSFER'])

        # Create a DataFrame with the input data
        data = pd.DataFrame({
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest],
            'isFlaggedFraud': [isFlaggedFraud],
            'CASH_OUT': [CASH_OUT],
            'DEBIT': [DEBIT],
            'NOTYPE': [NOTYPE],
            'PAYMENT': [PAYMENT],
            'TRANSFER': [TRANSFER]})
        
        # Make prediction using the loaded model
        prediction = model.predict(data)

        # Display prediction result
        if prediction[0] == 1:
            prediction_text = "Fraudulent Transaction Detected"
        else:
            prediction_text = "Non-Fraudulent Transaction"

        return render_template("index.html", prediction_text=prediction_text)
    else:
        return "Invalid request method"

# Run the app if executed directly
if __name__ == "__main__":
    app.run()
