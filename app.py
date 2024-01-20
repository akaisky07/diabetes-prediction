from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the form
            pregnancies = int(request.form['Pregnancies'])
            glucose = int(request.form['Glucose'])
            blood_pressure = int(request.form['BloodPressure'])
            skin_thickness = int(request.form['SkinThickness'])
            insulin = int(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            dpf = float(request.form['DiabetesPedigreeFunction'])
            age = int(request.form['Age'])

            # Make prediction
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            prediction = model.predict(input_data)

            return render_template('index.html', prediction=int(prediction[0]))

        except Exception as e:
            return render_template('index.html', error="Invalid input. Please check your input values.")

if __name__ == '__main__':
    app.run(debug=True)

