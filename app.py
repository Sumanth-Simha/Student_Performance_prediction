from flask import Flask, render_template, request
import joblib
import numpy as np
import os
# Load model
model = joblib.load('student_model.pkl')

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        studytime = float(request.form['studytime'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])

        # Create input array
        input_data = np.array([[studytime, failures, absences, G1, G2]])

        # Prediction
        prediction = model.predict(input_data)
        result = "PASS" if prediction[0] == 1 else "FAIL"

        return render_template('index.html', prediction_text=f"Prediction: The student will {result}")
    except:
        return render_template('index.html', prediction_text="Please check your inputs!")

if __name__ == '__main__':
    
    port = int(os.environ.get("PORT", 5000))  # Use Render's port or default to 5000
    app.run(host='0.0.0.0', port=port)
