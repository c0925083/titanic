import streamlit as st
import pickle
import numpy as np

# Load model & scaler
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Titanic Survival Prediction")

# User input
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.radio("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=30)
Fare = st.number_input("Fare", min_value=0.0, value=10.0)
Embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Convert inputs
Sex = 0 if Sex == "Male" else 1
Embarked = {"C": 0, "Q": 1, "S": 2}[Embarked]

if st.button("Predict"):
    input_data = np.array([[Pclass, Sex, Age, Fare, Embarked]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    result = "Survived" if prediction[0] == 1 else "Not Survived"
    
    st.write(f"Prediction: **{result}**")
 