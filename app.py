import streamlit as st
import pandas as pd
import joblib

try:
    preprocessor = joblib.load('model/preprocessor.joblib')
    best_model = joblib.load('model/fine_tunned_xgbclassifier.joblib')

except Exception as e:
    print('model or preprocessor file not found')
    st.stop()

st.title('Stroke Prediciton')
st.header('Enter Patient Details')

age = st.number_input('Age', min_value = 0.08, max_value = 82.0)
gender = st.selectbox('Gender', options = ['Male', 'Female'])
hypertension = st.selectbox('Hypertension', options = [0, 1])
heart_disease = st.selectbox('Heart Disease', options = [0, 1])
ever_married = st.selectbox('Ever Married', options = ['Yes', 'No'])
residence_type = st.selectbox('Residence Type', options = ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level',min_value = 55.12 , max_value = 271.74)
bmi = st.number_input('BMI', min_value = 10.3 , max_value = 97.6)
work_type = st.selectbox('Work Type', options = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
smoking_status = st.selectbox('Smoking Status', options=["never smoked", "formerly smoked", "smokes", "Unknown"])

predict_button = st.button('Predict')

if predict_button:
    data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'work_type': [work_type],
        'smoking_status': [smoking_status]
    })
    gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
    ever_married_mapping = {'No': 0, 'Yes': 1}
    residence_type_mapping = {'Rural': 0, 'Urban': 1}
    try:
        data['gender'] = data['gender'].map(gender_mapping)
        data['ever_married'] = data['ever_married'].map(ever_married_mapping)
        data['Residence_type'] = data['Residence_type'].map(residence_type_mapping)

        if data[['gender', 'ever_married', 'Residence_type']].isnull().any().any():
            raise ValueError("Invalid input value for gender, ever_married, or Residence_type")
        
    except Exception as e:
        st.error(f'Error encoding categorical variable: {e}')
        st.stop()


    data['work_type_Never_worked'] = (data['work_type'] == 'Never_worked').astype(int)
    data['work_type_Private'] = (data['work_type'] == 'Private').astype(int)
    data['work_type_Self-employed'] = (data['work_type'] == 'Self-employed').astype(int)
    data['work_type_children'] = (data['work_type'] == 'children').astype(int)

    data['smoking_status_formerly smoked'] = (data['smoking_status'] == 'formerly smoked').astype(int)
    data['smoking_status_smokes'] = (data['smoking_status'] == 'smokes').astype(int)
    data['smoking_status_unknown'] = (data['smoking_status'] == 'Unknown').astype(int)
    data['smoking_status_never smoked'] = (data['smoking_status'] == 'never smoked').astype(int)

    data['age_hypertension'] = data['age'] * data['hypertension']
    data['age_heart_disease'] = data['age'] * data['heart_disease']

    data = data.drop(['work_type','smoking_status'], axis = 1)

    expected_columns = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married','Residence_type',
        'avg_glucose_level', 'bmi', 'work_type_Never_worked',
        'work_type_Private', 'work_type_Self-employed', 'work_type_children',
        'smoking_status_formerly smoked', 'smoking_status_never smoked',
        'smoking_status_smokes', 'age_hypertension', 'age_heart_disease'
    ]
    data = data[expected_columns]

    try:
        data_transformed = preprocessor.transform(data)
    except Exception as e:
        st.error(f'Error applying preprocessor: {e}')
        st.stop()

    try:
        stroke_prob = best_model.predict_proba(data_transformed)[0,1] * 100
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.stop()

    st.subheader('Prediction Results')
    st.write(f'**Stroke Risk**: {stroke_prob:.2f}%')
    if stroke_prob > 50:
        st.warning('High stroke risk detected! Please consult a healthcare professional.')
    else:
        st.success('Low stroke risk predicted. Maintain a healthy lifestyle!')