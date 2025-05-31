import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

try:
    preprocessor = joblib.load('..\\model\\preprocessor.joblib')
    best_model = joblib.load('..\\model\\fine_tunned_xgbclassifier.joblib')
except Exception as e:
    print('model or preprocessor file not found')
st.title('Stroke Prediciton')
st.header('Enter Patient Details')

age = st.number_input('Age', min_value = 0.08, max_value = 82.0)
gender = st.selectbox('Gender', options = ['Male', 'Female'])
hypertension = st.selectbox('Hypertension', options = [0, 1])
heart_disease = st.selectbox('Heart Disease', options = [0, 1])
ever_married = st.selectbox('Ever Married', options = ['yes', 'no'])
residence_type = st.selectbox('Residence Type', options = ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level',min_value = 55.12 , max_value = 271.74)
bmi = st.number_input('BMI', min_value = 10.3 , max_value = 97.6)
work_type = st.selectbox('Work Type', options = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
smoking_status = st.selectbox('Smoking Status', options=["never smoked", "formerly smoked", "smokes", "Unknown"])

predict_buttom = st.button('Predict')

if predict_buttom:
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'work_type': work_type,
        'smoking_status': smoking_status
    })