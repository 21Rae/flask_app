from flask import Flask, request, render_template
import joblib
import pickle

# Save an object
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load the object
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

import numpy as np

app = Flask(__name__)
#Load the model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        try:
            # Get users inpute
            feature1 = float(request.form["feature1"])
            feature2 = float(request.form["feature2"])

            # Make Prediction
            input_data = np.array([[feature1, feature2]])
            prediction = model.predict(input_data)[0]

            return render_template("index.html", prediction=prediction)
        except Exception as e:
            return render_template("index.html", error= "Invalid input. Please enter a number")
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
