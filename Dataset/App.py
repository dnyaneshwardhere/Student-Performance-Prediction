from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained models
classification_model = pickle.load(open("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/classification_model.pkl", "rb"))
regression_model = pickle.load(open("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/regression_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        student_name = request.form["name"]
        prn_number = request.form["prn"]
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        assignment_scores = float(request.form["assignment_scores"])
        last_sem_percentage = float(request.form["last_sem_percentage"])
        mobile_screen_time = float(request.form["mobile_screen_time"])
        sleep_hours = float(request.form["sleep_hours"])

        # Prepare input features
        input_features = np.array([[study_hours, attendance, assignment_scores, last_sem_percentage, mobile_screen_time, sleep_hours]])

        # Predict percentage using regression model
        predicted_percentage = regression_model.predict(input_features)[0]

        # Predict pass/fail using classification model
        pass_fail_prediction = classification_model.predict(input_features)[0]
        pass_fail_status = "Pass" if pass_fail_prediction == 1 else "Fail"

        return render_template("result.html", name=student_name, prn=prn_number, 
                               predicted_percentage=predicted_percentage, pass_fail_status=pass_fail_status)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
