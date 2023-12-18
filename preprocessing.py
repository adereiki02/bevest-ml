import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def normalize_data(age, gender, income, education, marital_status, number_of_children, home_ownership):
    # Load the saved scaler using joblib
    scaler = joblib.load('normalization_model.joblib')
    data = np.array([[age, gender, income, education, marital_status, number_of_children, home_ownership]])
    # Normalize the data using Joblib
    normalized_data = scaler.transform(data)
    return normalized_data

def preprocess_data(age, gender, income, education, marital_status, number_of_children, home_ownership):
    normalized_data = normalize_data(age, gender, income, education, marital_status, number_of_children, home_ownership)
    return normalized_data