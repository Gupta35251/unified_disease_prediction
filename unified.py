import streamlit as st
import pickle
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Unified Disease Prediction Platform", layout="centered")

# ---------------- UI HEADER ----------------
st.title("🩺 Unified Disease Prediction Platform")
st.caption("College Mini Project | For Educational Use Only")
st.caption("")
st.caption("Made by 𝘗𝘳𝘪𝘯𝘤𝘦 𝘎𝘶𝘱𝘵𝘢")


st.sidebar.title("🔍 Select Prediction")
prediction_type = st.sidebar.selectbox(
    "Choose a model",
    ["Diabetes", "Heart Disease", "Breast Cancer"]
)
# prediction_type = st.sidebar.radio(
#     "Choose a model",
#     ["Diabetes", "Heart Disease", "Breast Cancer"]
# )





# ==================================================
# DIABETES PREDICTION (MODEL ONLY)
# ==================================================
if prediction_type == "Diabetes":
    st.subheader("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Number of Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("Body Mass Index (BMI)")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        scaler = pickle.load(open(
            "Models and preprocessings/diabetes_prediction_scaler.pkl", "rb"
        ))
        model = pickle.load(open(
            "Models and preprocessings/diabetes_prediction_model.pkl", "rb"
        ))

        input_data = np.array([[pregnancies, glucose, bp, skin,
                                insulin, bmi, dpf, age]])
        scaled_data = scaler.transform(input_data)

        result = model.predict(scaled_data)

        prediction_text = "Diabetes Detected" if result[0] == 1 else "No Diabetes Detected"
        st.success(prediction_text) if result[0] == 0 else st.error(prediction_text)

        # -------- DOWNLOAD REPORT --------
        report = f"""
        DIABETES PREDICTION REPORT
        Date: {datetime.now()}

        Inputs:
        Pregnancies: {pregnancies}
        Glucose: {glucose}
        Blood Pressure: {bp}
        BMI: {bmi}
        Age: {age}

        Result: {prediction_text}

        Disclaimer: Educational use only.
        """

        st.download_button(
            "📄 Download Report",
            report,
            file_name="diabetes_report.txt"
        )

# ==================================================
# HEART DISEASE (SINGLE ENCODER + SCALER + MODEL)
# ==================================================
elif prediction_type == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex", ["male", "female"])
    cp = st.selectbox("Chest Pain Type", [
        "typical angina",
        "atypical angina",
        "non-anginal pain",
        "asymptomatic"
    ])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Serum Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG", [
        "normal",
        "st-t abnormality",
        "left ventricular hypertrophy"
    ])
    thalach = st.number_input("Maximum Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression")
    slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Number of Major Vessels (0–3)")
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

    if st.button("Predict Heart Disease"):
        encoder = pickle.load(open(
            "Models and preprocessings/heart_disease_prediction_encoder.pkl", "rb"
        ))
        scaler = pickle.load(open(
            "Models and preprocessings/heart_disease_prediction_scaler.pkl", "rb"
        ))
        model = pickle.load(open(
            "Models and preprocessings/heart_disease_prediction_model.pkl", "rb"
        ))

        # cp_encoder = pickle.load(open("Models and preprocessings/heart_disease_prediction_cp_encoder.pkl","rb"))
        # restecg_encoder = pickle.load(open("Models and preprocessings/heart_disease_prediction_restecg_encoder.pkl","rb"))
        # slope_encoder = pickle.load(open("Models and preprocessings/heart_disease_prediction_slope_encoder.pkl","rb"))
        # # thalach_encoder = pickle.load(open("Models and preprocessings/heart_disease_prediction_thalach_encoder.pkl","rb"))

        sex = 1 if sex == "male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        # sex = 1 if sex == "male" else 0

        cp_map = {
            "typical angina": 0,
            "atypical angina": 1,
            "non-anginal pain": 2,
            "asymptomatic": 3
        }
        cp = cp_map[cp]

        restecg_map = {
            "normal": 0,
            "st-t abnormality": 1,
            "left ventricular hypertrophy": 2
        }
        restecg = restecg_map[restecg]

        slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
        slope = slope_map[slope]

        thal_map = {"normal": 1, "fixed defect": 2, "reversible defect": 3}
        thal = thal_map[thal]

        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        # cp = cp_encoder.transform([cp])[0]
        # restecg = restecg_encoder.transform([restecg])[0]
        # slope = slope_encoder.transform([slope])[0]
        # # thalach = thalach_encoder.transform([thalach])
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])
        scaled = scaler.transform(input_data)
        result = model.predict(scaled)[0]

        level_map = {
            0 : "No heart Disease",
            1 : "Mild Heart Disease",
            2 : "Severe heart Disease"
        }

        prediction_text = level_map[result]

        if result == 0:
            st.success(f"🟢 {prediction_text}")
        elif result == 1:
            st.warning(f"🟡 {prediction_text}")
        else:
            st.error(f"🔴 {prediction_text}")

        # prediction_text = "Heart Disease Detected" if result[0] == 1 else "No Heart Disease"
        # st.error(prediction_text) if result[0] == 1 else st.success(prediction_text)
        st.info(f"The level of the disease prediction is : {result}")
        # -------- DOWNLOAD REPORT --------
        report = f"""
        HEART DISEASE PREDICTION REPORT
        Date: {datetime.now()}

        Age: {age}
        Sex: {sex}
        Chest Pain: {cp}
        Cholesterol: {chol}

        Result: {prediction_text}

        Disclaimer: Educational use only.
        """

        st.download_button(
            "📄 Download Report",
            report,
            file_name="heart_disease_report.txt"
        )

# ==================================================
# BREAST CANCER (MODEL ONLY)
# ==================================================
elif prediction_type == "Breast Cancer":
    st.subheader("🧬 Breast Cancer Prediction")

    inputs = []
    features = [
        "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area",
        "Mean Smoothness", "Mean Compactness", "Mean Concavity",
        "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
        "Radius Error", "Texture Error", "Perimeter Error", "Area Error",
        "Smoothness Error", "Compactness Error", "Concavity Error",
        "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
        "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area",
        "Worst Smoothness", "Worst Compactness", "Worst Concavity",
        "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
    ]

    for feature in features:
        inputs.append(st.number_input(feature))

    if st.button("Predict Breast Cancer"):
        model = pickle.load(open(
            "Models and preprocessings/breast_cancer_prediction_model.pkl", "rb"
        ))

        data = np.array([inputs])
        result = model.predict(data)

        prediction_text = "Breast Cancer Detected" if result[0] == 1 else "No Breast Cancer"
        st.error(prediction_text) if result[0] == 1 else st.success(prediction_text)

        # -------- DOWNLOAD REPORT --------
        report = f"""
        BREAST CANCER PREDICTION REPORT
        Date: {datetime.now()}

        Result: {prediction_text}

        Disclaimer: Educational use only.
        """

        st.download_button(
            "📄 Download Report",
            report,
            file_name="breast_cancer_report.txt"
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ This system is for academic and research purposes only.")
st.markdown("---")
