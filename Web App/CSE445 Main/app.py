from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import requests

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("best_model_CatBoost.pkl")
scaler = joblib.load("scaler.pkl")

# Define Label Encoding Dictionary
label_encodings = {
    "Gender": {"Male": 0, "Female": 1}
}

# Define Target Decoding
target_decoding = {
    2: "Obesity Type I",
    4: "Obesity Type III",
    3: "Obesity Type II",
    5: "Overweight Level I",
    6: "Overweight Level II",
    1: "Normal Weight",
    0: "Insufficient Weight"
}

# Ollama API details
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

def get_diet_chart(obesity_level):
    """Send obesity level to Ollama and get diet chart response."""
    prompt = f"Generate a detailed diet chart for a person with {obesity_level}. Include meal recommendations and nutritional advice."
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        response_json = response.json()
        return response_json.get("response", "No diet chart available.")
    except Exception as e:
        return f"Error fetching diet chart: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = request.json

        # Convert input into DataFrame
        df = pd.DataFrame([input_data])

        # Apply Label Encoding
        for col, mapping in label_encodings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Select only the required features
        selected_features = ["Gender", "Age", "Height", "Weight",
                             "Frequency of consumption of vegetables", "BMR", "Nutritional_Score"]
        df = df[selected_features]

        # Convert to NumPy array
        X_new = df.values

        # Apply MinMax Scaling
        X_new_scaled = scaler.transform(X_new)

        # Make prediction
        y_pred_encoded = model.predict(X_new_scaled)

        # Convert encoded prediction back to actual label
        y_pred_label = target_decoding[int(y_pred_encoded[0])]

        # Get diet chart from Ollama
        diet_chart = get_diet_chart(y_pred_label)

        return jsonify({"prediction": y_pred_label, "diet_chart": diet_chart})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
