from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# ---------------------------
# Initialize Flask App
# ---------------------------
app = Flask(__name__)   # templates folder is auto-detected


# ---------------------------
# Load Model & Encoder
# ---------------------------
try:
    model = joblib.load("models/model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    print("✅ Model & Label Encoder loaded successfully!")
except Exception as e:
    print("❌ Failed to load model files:", e)


# ---------------------------
# Home Route — HTML UI
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")     # Make sure your file name matches


# ---------------------------
# Prediction Route (POST API)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert to array (order must match training)
        input_values = list(data.values())
        input_array = np.array(input_values).reshape(1, -1)

        # Predict encoded class
        pred_encoded = model.predict(input_array)[0]

        # Decode label (Low, Medium, High)
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return jsonify({
            "prediction": pred_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------
# Run App (only for local)
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
