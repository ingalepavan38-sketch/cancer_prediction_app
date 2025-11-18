from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load models
model = pickle.load(open("models/final_model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

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
