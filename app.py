from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model + label encoder
model = joblib.load("models/model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---------------------------
# HOME PAGE (User Interface)
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# PREDICTION API
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        input_data = np.array(list(data.values())).reshape(1, -1)
        pred_encoded = model.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return jsonify({"prediction": pred_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
