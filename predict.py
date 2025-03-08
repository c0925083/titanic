import pickle
import numpy as np

# Load model & scaler
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_survival(Pclass, Sex, Age, Fare, Embarked):
    # Convert input to a NumPy array
    input_data = np.array([[Pclass, Sex, Age, Fare, Embarked]])
    
    # Standardize input
    input_data = scaler.transform(input_data)

    # Predict survival
    prediction = model.predict(input_data)
    return "Survived" if prediction[0] == 1 else "Not Survived"

print(predict_survival(3, 0, 25, 7.25, 2))  # 3rd class male, 25 years old, low fare
