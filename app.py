from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Load the trained model
model = joblib.load('student_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        studytime = float(request.form['studytime'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])

        # Create input array
        input_data = np.array([[studytime, failures, absences, G1, G2]])

        # Make prediction
        prediction = model.predict(input_data)
        result = "PASS" if prediction[0] == 1 else "FAIL"

        return render_template('index.html', prediction_text=f"Prediction: The student will {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

# Port binding for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
