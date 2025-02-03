import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # HTML form for input

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        data = [float(x) for x in request.form.values()]
        input_data = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
