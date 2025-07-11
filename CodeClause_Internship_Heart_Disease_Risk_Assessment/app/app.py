import streamlit as st
import pandas as pd
import joblib

# 🔁 Load model
model = joblib.load(r"C:\CodeClause_Internship_Heart_Disease_Risk_Assessment\model\xgb_heart_model.pkl")

# 🧾 Input form
st.title("❤️ Heart Disease Risk Predictor")

def user_input():
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        age = age_years * 365
    with col2:
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 2 else "Female")

    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
    with col4:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

    col5, col6 = st.columns(2)
    with col5:
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
    with col6:
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)

    col7, col8 = st.columns(2)
    with col7:
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above"][x-1])
    with col8:
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above"][x-1])

    col9, col10 = st.columns(2)
    with col9:
        smoke = st.selectbox("Smokes?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col10:
        alco = st.selectbox("Alcohol Intake?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    col11, col12 = st.columns(2)
    with col11:
        active = st.selectbox("Physically Active?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")


    data = pd.DataFrame([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]],
                        columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                                 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
    return data

# 🧠 Prediction
input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Cardiovascular Disease ({prob*100:.2f}% confidence)")
    else:
        st.success(f"✅ Low Risk ({(1 - prob)*100:.2f}% confidence)")
