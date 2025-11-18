from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import sys

def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading {path}: {e}")
        try:
            with open(path, "rb") as f:
                return pickle.load(f, encoding='latin1', fix_imports=True)
        except Exception as e2:
                    try:
            with open(path, "rb") as f:
                return pickle.load(f, fix_imports=True)
        except Exception as e3:
            print(f"Final attempt failed: {e3}")
            raise
app = Flask(__name__)

# Load models
try:
    model = load_model("models/final_model.pkl")
    label_encoder = load_model("models/label_encoder.pkl")
except Exception as e:
    print(f"Failed to load model files: {e}")
    sys.exit(1)
# All feature names
features = [
    "Air Pollution","Alcohol use","Dust Allergy","OccuPational Hazards",
    "Genetic Risk","chronic Lung Disease","Balanced Diet","Obesity",
    "Smoking","Passive Smoker","Chest Pain","Coughing of Blood",
    "Fatigue","Weight Loss","Shortness of Breath","Wheezing",
    "Swallowing Difficulty","Clubbing of Finger Nails","Frequent Cold",
    "Dry Cough","Snoring"
]

@app.route("/")
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    # Collect inputs
    data = {}

    data["Age"] = int(request.form["Age"])
    data["Gender"] = 1 if request.form["Gender"] == "Male" else 0

    for f in features:
        data[f] = int(request.form[f])

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    return render_template("result.html", prediction=label)

if __name__ == "__main__":
    app.run()
