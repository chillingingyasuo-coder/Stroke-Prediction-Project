import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

# تحميل الموديل والـ Scaler
model = XGBClassifier()
model.load_model('stroke_model.json')
scaler = pickle.load(open('scaler.sav', 'rb'))

st.title(" Stroke Prediction App")
st.write("أدخل بيانات المريض للتنبؤ باحتمالية الإصابة بالسكتة الدماغية")

# مدخلات المستخدم
age = st.slider("العمر (Age)", 1, 100, 45)
avg_glucose_level = st.number_input("مستوى الجلوكوز", value=90.0)
bmi = st.number_input("BMI", value=25.0)

gender = st.selectbox("الجنس", ["ذكر", "أنثى", "آخر"])
hypertension = st.selectbox("هل يعاني من ضغط الدم؟", ["لا", "نعم"])
heart_disease = st.selectbox("هل يعاني من أمراض القلب؟", ["لا", "نعم"])
ever_married = st.selectbox("هل سبق له الزواج؟", ["نعم", "لا"])
work_type = st.selectbox(
    "نوع العمل",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)
residence = st.selectbox("مكان الإقامة", ["Urban", "Rural"])
smoking = st.selectbox(
    "حالة التدخين",
    ["Unknown", "never smoked", "formerly smoked", "smokes"]
)

if st.button("Predict"):

    has_hypertension = 1 if hypertension == "نعم" else 0
    is_smoker = 1 if smoking == "smokes" else 0

    input_data = {
        "age": age,
        "hypertension": has_hypertension,
        "heart_disease": 1 if heart_disease == "نعم" else 0,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender_Male": 1 if gender == "ذكر" else 0,
        "gender_Other": 1 if gender == "آخر" else 0,
        "ever_married_Yes": 1 if ever_married == "نعم" else 0,
        "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
        "work_type_Private": 1 if work_type == "Private" else 0,
        "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
        "work_type_children": 1 if work_type == "children" else 0,
        "Residence_type_Urban": 1 if residence == "Urban" else 0,
        "smoking_status_formerly smoked": 1 if smoking == "formerly smoked" else 0,
        "smoking_status_never smoked": 1 if smoking == "never smoked" else 0,
        "smoking_status_smokes": is_smoker,
        "is_senior": 1 if age >= 60 else 0,
        "smokes_and_has_hypertension": 1 if (is_smoker == 1 and has_hypertension == 1) else 0,
        "is_obese": 1 if bmi >= 30 else 0,
        "is_diabetic": 1 if avg_glucose_level >= 126 else 0,
        "senior_hypertension": 1 if (age > 60 and has_hypertension == 1) else 0,
    }

    correct_order = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Male', 'gender_Other', 'ever_married_Yes',
        'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
        'Residence_type_Urban',
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
        'is_senior', 'smokes_and_has_hypertension', 'is_obese', 'is_diabetic', 'senior_hypertension'
    ]

    df_input = pd.DataFrame([input_data])[correct_order]


    scaled_data = scaler.transform(df_input)
    probability = model.predict_proba(scaled_data)[0][1]
    relative_risk = probability / 0.018

    st.write(f"احتمالية الإصابة بالسكتة الدماغية: **{probability:.2%}**")
    st.write(f"خطره أعلى من المعدل الطبيعي بـ **{relative_risk:.0f} مرة**")

    if probability >= 0.30:
        st.error(" خطر مرتفع جداً")
    elif probability >= 0.15:
        st.warning(" المريض معرض لخطر الإصابة بالسكتة الدماغية.")
    else:
        st.success(" خطر الإصابة منخفض.")