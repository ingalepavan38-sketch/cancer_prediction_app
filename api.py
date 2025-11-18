from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model/xgb_model.joblib")
le = joblib.load("model/label_encoder.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        pred_encoded = model.predict(df)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        return jsonify({
            "prediction_encoded": int(pred_encoded),
            "prediction_label": pred_label
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()

