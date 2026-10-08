import streamlit as st
import seaborn as sns
import joblib
import pandas as pd

# streamlit page config
st.set_page_config(
    page_title="Chronic Heart Disease Dashboard",  # the page title shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)

# 2. Set up UI
st.title("10 Year risk of coronary heart disease (CHD) prediction")
st.header("User Input")

# Define input widgets
user_male            = st.radio("male            ",["Male","Female"])
user_age             = st.slider("age            ",min_value=0, max_value=100, value=40)
user_education       = st.radio("education Years ",[1,2,3,4])
user_cigsPerDay      = st.slider("cigsPerDay      ",min_value=0, max_value=100, value=30)
user_totChol         = st.slider("totChol         ",min_value=0, max_value=1000, value=100)
user_sysBP           = st.slider("sysBP           ",min_value=70, max_value=500, value=100)
user_diaBP           = st.slider("diaBP           ",min_value=70, max_value=500, value=100)
user_BMI             = st.slider("BMI             ",min_value=70, max_value=500, value=100)
user_heartRate       = st.slider("heartRate       ",min_value=70, max_value=500, value=100)
user_glucose         = st.slider("glucose         ",min_value=70, max_value=500, value=100)
user_currentSmoker   = st.checkbox("currentSmoker   ")
user_BPMeds          = st.checkbox("BPMeds          ")
user_prevalentStroke = st.checkbox("prevalentStroke ")
user_prevalentHyp    = st.checkbox("prevalentHyp    ")
user_diabetes        = st.checkbox("diabetes        ")

# 3. Handle Prediction on Button Click
if st.button("Predict"):
   # Prepare data for the model
   user_data = pd.DataFrame({
        "male": [user_male],
        "age ": [user_age],
        "education": [user_education],
        "currentSmoker": [1 if user_currentSmoker else 0],
        "cigsPerDay": [1 if user_cigsPerDay else 0],
        "BPMeds": [1 if user_BPMeds else 0],
        "prevalentStroke": [1 if user_prevalentStroke else 0],
        "prevalentHyp": [1 if user_prevalentHyp else 0],
        "diabetes": [1 if user_diabetes else 0],
        "totChol": [user_totChol],
        "sysBP": [user_sysBP],
        "diaBP": [user_diaBP],
        "BMI": [user_BMI],
        "heartRate": [user_heartRate],
        "glucose": [user_glucose]
   })

   # Make prediction
   #prediction = CHD_prediction(input_data)
   prediction = [1]

    # Display result
   if prediction[0] == 1:
       st.error("At Risk of coronary heart disease in next 10 years")
   else:
       st.success("Not At risk of coronary heart disease in next 10 years")

