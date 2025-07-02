import numpy as np
from flask import Flask, render_template, request
from model import predict_crop_and_profit  # Import the function from model.py

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input features from the form
        nitrogen = float(request.form["Nitrogen"])
        phosphorus = float(request.form["Phosphorus"])
        potassium = float(request.form["Potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["pH"])
        rainfall = float(request.form["rainfall"])
        total_land = float(request.form["total_land"])
        main_crop_land = float(request.form["main_crop_land"])

        # Get crop & spacing prediction
        result_text = predict_crop_and_profit(nitrogen, phosphorus, potassium, temperature, rainfall, ph, humidity, total_land, main_crop_land)

        return render_template("index.html", prediction_text=result_text)
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
    

if __name__ == "__main__":
    app.run(debug=True)
