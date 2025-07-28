# app.py
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('models/logreg_model.pkl')

# Page title
st.title("🧠 Diabetes Prediction App")

# Input form
st.sidebar.header("Patient Data Input")

def user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 25.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.sidebar.slider('Age', 21, 100, 33)
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    return features

input_data = user_input()

# Predict
prediction = model.predict(input_data)[0]
result = "Diabetic" if prediction == 1 else "Non-diabetic"

# Display
st.subheader("Prediction Result:")
st.write(f"The patient is **{result}**.")
