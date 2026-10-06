import streamlit as st
import seaborn as sns
import joblib
import pandas as pd

# streamlit page config
st.set_page_config(
    page_title="Iris Dashboard",  # the page title shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)
# add page title
# load data
df = sns.load_dataset('iris')
# get column and species names
attributes = df.columns[:-1].tolist()
species = df['species'].unique().tolist()




st.write("Loading pre-trained model")

# 2. Set up UI
st.title("Stroke Risk Prediction")
st.header("User Input")

# Define input widgets
user_age = st.slider("Age", min_value=0, max_value=100, value=30)
user_gender = st.radio("Gender", ["Male", "Female"])
user_bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
user_avg_glucose = st.number_input("Avg Glucose Level", min_value=0.0, value=80.0)
user_hypertension = st.checkbox("Hypertension")
user_heart_disease = st.checkbox("Heart Disease")

# 3. Handle Prediction on Button Click
if st.button("Predict"):
    # Prepare data for the model
    user_data = pd.DataFrame({
        "age": [user_age],
        "gender": [user_gender],
        "bmi": [user_bmi],
        "avg_glucose_level": [user_avg_glucose],
        "hypertension": [1 if user_hypertension else 0],
        "heart_disease": [1 if user_heart_disease else 0]
    })

    # Make prediction
   # prediction = loaded_model.predict(user_data)
    st.write("making prediction")

    prediction = [1]
    
    # Display result
    if prediction[0] == 1:
        st.error("At Risk")
    else:
        st.success("Not At Risk")

