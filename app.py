# Important Modules
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='templates')


# Paths
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Serve uploaded files
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Routes
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

# Prediction function
def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)

    if size == 8:  # Diabetes
        loaded_model = pickle.load(open('diabetes_best.pkl', 'rb'))
        result = loaded_model.predict(to_predict)

    elif size == 30:  # Cancer
        loaded_model = pickle.load(open('cancer_best.pkl', 'rb'))
        result = loaded_model.predict(to_predict)

    elif size == 12:  # Kidney
        loaded_model = pickle.load(open("kidney_best.pkl", 'rb'))
        result = loaded_model.predict(to_predict)

    elif size == 11:  # Heart
        loaded_model = pickle.load(open("heart_best.pkl", 'rb'))
        result = loaded_model.predict(to_predict)

    return result[0], size

# Result route
@app.route('/result', methods=["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))

        if len(to_predict_list) == 30:  # Cancer
            Result, size = ValuePredictor(to_predict_list, 30)
        elif len(to_predict_list) == 8:  # Diabetes
            Result, size = ValuePredictor(to_predict_list, 8)
        elif len(to_predict_list) == 12:  # Kidney
            Result, size = ValuePredictor(to_predict_list, 12)
        elif len(to_predict_list) == 11:  # Heart
            Result, size = ValuePredictor(to_predict_list, 11)

    # Disease-specific messages
    if int(Result) == 1 and size == 8:
        prediction = 'Sorry! You are suffering from Diabetes disease'
    elif int(Result) == 1 and size == 11:
        prediction = 'Sorry! You are suffering from Heart disease'
    elif int(Result) == 1 and size == 12:
        prediction = 'Sorry! You are suffering from Kidney disease'
    elif int(Result) == 1 and size == 30:
        prediction = 'Sorry! You are suffering from Cancer disease'
    else:
        prediction = 'Congrats! You are Healthy'

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
