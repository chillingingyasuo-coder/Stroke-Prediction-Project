import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. تحميل النموذج والـ Scaler
try:
    model = pickle.load(open('stroke_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except Exception as e:
    st.error(f"خطأ في تحميل ملفات الموديل: {e}")

# 2. واجهة التطبيق وحقول الإدخال
st.title("🧠 نظام التنبؤ بالسكتة الدماغية (Cerebral Stroke Prediction)")
st.write("الرجاء إدخال بيانات المريض للتنبؤ باحتمالية الإصابة:")

age = st.number_input("العمر (Age)", min_value=0, max_value=120, value=82)
avg_glucose_level = st.number_input("مستوى السكر في الدم (Avg Glucose Level)", min_value=0.0, value=240.0)
bmi = st.number_input("مؤشر كتلة الجسم (BMI)", min_value=0.0, value=31.0)

gender = st.selectbox("الجنس (Gender)", ["Male", "Female", "Other"])
hypertension = st.selectbox("هل يعاني من ضغط الدم؟ (Hypertension)", [0, 1], index=1)
heart_disease = st.selectbox("هل يعاني من مرض في القلب؟ (Heart Disease)", [0, 1], index=1)
ever_married = st.selectbox("هل سبق له الزواج؟ (Ever Married)", ["Yes", "No"], index=0)

work_type = st.selectbox("نوع العمل (Work Type)", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], index=1)
Residence_type = st.selectbox("مكان الإقامة (Residence Type)", ["Urban", "Rural"], index=0)
smoking_status = st.selectbox("حالة التدخين (Smoking Status)", ["formerly smoked", "never smoked", "smokes", "Unknown"], index=0)

# 3. بناء قاموس الميزات
raw_features = {
    'id': 0, 'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose_level, 'bmi': bmi,
    'is_senior': 1 if age > 60 else 0,
    'is_obese': 1 if bmi > 30 else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Other': 0,
    'ever_married_Yes': 1 if ever_married == "Yes" else 0,
    'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
    'work_type_Private': 1 if work_type == "Private" else 0,
    'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
    'work_type_children': 1 if work_type == "children" else 0,
    'Residence_type_Urban': 1 if Residence_type == "Urban" else 0,
    'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
    'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
    'smoking_status_smokes': 1 if smoking_status == "smokes" else 0
}

input_data = pd.DataFrame([raw_features])

# 4. معالجة البيانات والتنبؤ
if st.button("تنبؤ باحتمالية الإصابة"):
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = list(scaler.feature_names_in_)
        final_input = pd.DataFrame(columns=expected_features)
        for col in expected_features:
            final_input[col] = input_data[col] if col in input_data.columns else 0
        
        input_scaled = scaler.transform(final_input)
        
        try:
            probabilities = model.predict_proba(input_scaled)
            stroke_prob = probabilities[0][1] 
        except AttributeError:
            stroke_prob = 0.0
            st.warning("الموديل لا يدعم predict_proba")
            
        st.write("---")
        st.info(f"📊 احتمالية الإصابة الحقيقية المستخرجة من الموديل: {stroke_prob * 100:.2f}%")
        
        if stroke_prob > 0.10: 
            st.error("⚠️ تحذير: هذا المريض معرض لخطر الإصابة بسكتة دماغية (High Risk).")
        else:
            st.success("✅ المريض غير معرض لخطر الإصابة بسكتة دماغية حالياً (Low Risk).")
    else:
        st.error("الـ Scaler المحفوظ لا يحتوي على أسماء الميزات.")