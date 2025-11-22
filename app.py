import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =======================================
# LOAD MODEL, SCALER, ENCODER COLUMNS
# =======================================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =======================================
# CSS STYLING FOR MODERN UI
# =======================================
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #2b5876;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #4a4a4a;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        div.stButton > button {
            background-color: #2b5876;
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
            font-size: 18px;
            width: 100%;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #4e4376;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Employee Attrition Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter employee information below for prediction</p>", unsafe_allow_html=True)

# =======================================
# SECTION 1 — PERSONAL DETAILS
# =======================================
st.markdown("<div class='card'><h3> Personal Details</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", 18, 60, 30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    EducationField = st.selectbox("Education Field", 
                                  ["Life Sciences", "Medical", "Marketing",
                                   "Technical Degree", "Human Resources", "Other"])
    DistanceFromHome = st.number_input("Distance From Home (km)", 1, 50, 5)

st.markdown("</div>", unsafe_allow_html=True)

# =======================================
# SECTION 2 — JOB DETAILS
# =======================================
st.markdown("<div class='card'><h3> Job Details</h3>", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    BusinessTravel = st.selectbox("Business Travel", 
                                  ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    Department = st.selectbox("Department", 
                               ["Sales", "Research & Development", "Human Resources"])
    JobRole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Research Director", "Human Resources", "Sales Representative"
    ])
    JobLevel = st.number_input("Job Level", 1, 5, 1)

with col4:
    JobInvolvement = st.number_input("Job Involvement (1–4)", 1, 4, 3)
    JobSatisfaction = st.number_input("Job Satisfaction (1–4)", 1, 4, 3)
    EnvironmentSatisfaction = st.number_input("Environment Satisfaction (1–4)", 1, 4, 3)
    RelationshipSatisfaction = st.number_input("Relationship Satisfaction (1–4)", 1, 4, 3)
    WorkLifeBalance = st.number_input("Work Life Balance (1–4)", 1, 4, 3)

st.markdown("</div>", unsafe_allow_html=True)

# =======================================
# SECTION 3 — COMPENSATION
# =======================================
st.markdown("<div class='card'><h3> Compensation</h3>", unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
    HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)
    MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)

with col6:
    MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 20000)
    PercentSalaryHike = st.number_input("Percent Salary Hike", 1, 25, 10)
    OverTime = st.selectbox("Over Time", ["Yes", "No"])

st.markdown("</div>", unsafe_allow_html=True)

# =======================================
# SECTION 4 — EXPERIENCE
# =======================================
st.markdown("<div class='card'><h3> Experience</h3>", unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
    YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)
    YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, 2)

with col8:
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
    YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 2)
    NumCompaniesWorked = st.number_input("Num Companies Worked", 0, 10, 1)
    TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 2)

st.markdown("</div>", unsafe_allow_html=True)

# =======================================
# BUILD INPUT DF
# =======================================
input_data = pd.DataFrame({
    "Age": [Age],
    "Gender": [Gender],
    "MaritalStatus": [MaritalStatus],
    "EducationField": [EducationField],
    "DistanceFromHome": [DistanceFromHome],

    "BusinessTravel": [BusinessTravel],
    "Department": [Department],
    "JobRole": [JobRole],
    "JobLevel": [JobLevel],
    "JobInvolvement": [JobInvolvement],
    "JobSatisfaction": [JobSatisfaction],
    "EnvironmentSatisfaction": [EnvironmentSatisfaction],
    "RelationshipSatisfaction": [RelationshipSatisfaction],
    "WorkLifeBalance": [WorkLifeBalance],

    "DailyRate": [DailyRate],
    "HourlyRate": [HourlyRate],
    "MonthlyIncome": [MonthlyIncome],
    "MonthlyRate": [MonthlyRate],
    "PercentSalaryHike": [PercentSalaryHike],
    "OverTime": [OverTime],

    "TotalWorkingYears": [TotalWorkingYears],
    "YearsAtCompany": [YearsAtCompany],
    "YearsInCurrentRole": [YearsInCurrentRole],
    "YearsSinceLastPromotion": [YearsSinceLastPromotion],
    "YearsWithCurrManager": [YearsWithCurrManager],
    "NumCompaniesWorked": [NumCompaniesWorked],
    "TrainingTimesLastYear": [TrainingTimesLastYear],
})

# =======================================
# ONE-HOT ENCODE + FIX MISSING COLUMNS
# =======================================
input_encoded = pd.get_dummies(input_data)

for col in columns:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[columns]

# SCALE
input_scaled = scaler.transform(input_encoded)

# =======================================
# PREDICT BUTTON
# =======================================
if st.button("Predict Attrition"):
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.markdown("""
            <div style='padding:20px; 
                        background-color:#ffe6e6; 
                        border-left:6px solid #ff4d4d; 
                        border-radius:10px; 
                        font-size:18px; 
                        font-weight:500;'>
                <strong style="color:#cc0000;">High Attrition Risk</strong><br>
                This employee is <strong style='color:#cc0000;'>LIKELY</strong> to leave the company.
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <div style='padding:20px; 
                        background-color:#e6ffe6; 
                        border-left:6px solid #00b33c; 
                        border-radius:10px; 
                        font-size:18px; 
                        font-weight:500;'>
                <strong style="color:#009933;">Low Attrition Risk</strong><br>
                This employee is <strong style='color:#009933;'>NOT</strong> likely to leave the company.
            </div>
        """, unsafe_allow_html=True)

