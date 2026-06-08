import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Car Insurance Claim Predictor", layout="wide", page_icon="🚗")

# Styling to make it look premium
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #125b8a;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Car Insurance Claim Prediction Dashboard")
st.markdown("Enter the driver and vehicle details below to predict the likelihood of a claim and the estimated cost.")

with st.sidebar:
    st.header("Driver Demographics")
    age = st.slider("Age", 16, 100, 40)
    gender = st.selectbox("Gender", ["M", "F"])
    married = st.selectbox("Married", ["Yes", "No"])
    single_parent = st.selectbox("Single Parent", ["Yes", "No"])
    num_of_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
    highest_education = st.selectbox("Highest Education", ['<High School', 'High School', 'Bachelors', 'Masters', 'PhD'])
    occupation = st.selectbox("Occupation", ['Professional', 'Blue Collar', 'Manager', 'Clerical', 'Doctor', 'Lawyer', 'Home Maker', 'Student'])
    
    st.header("Financial & Location")
    income = st.number_input("Income ($)", min_value=0.0, value=50000.0)
    years_job_held_for = st.number_input("Years Job Held For", min_value=0.0, max_value=50.0, value=5.0)
    value_of_home = st.number_input("Value of Home ($)", min_value=0.0, value=200000.0)
    address_type = st.selectbox("Address Type", ["Highly Urban/ Urban", "Highly Rural/ Rural"])
    
col1, col2 = st.columns(2)
with col1:
    st.subheader("Vehicle Details")
    vehicle_type = st.selectbox("Vehicle Type", ["Minivan", "Van", "SUV", "Sports Car", "Panel Truck", "Pickup"])
    vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0.0, max_value=50.0, value=5.0)
    vehicle_value = st.number_input("Vehicle Value ($)", min_value=0.0, value=15000.0)
    type_of_use = st.selectbox("Type of Use", ["Private", "Commercial"])
    commute_dist = st.number_input("Commute Distance (miles)", min_value=0.0, value=10.0)

with col2:
    st.subheader("Insurance & Driving History")
    policy_tenure = st.number_input("Policy Tenure (Years)", min_value=0.0, max_value=50.0, value=5.0)
    num_young_drivers = st.number_input("Number of Young Drivers", min_value=0.0, max_value=10.0, value=0.0)
    five_year_num_of_claims = st.number_input("Number of Claims (Last 5 Years)", min_value=0.0, max_value=20.0, value=0.0)
    five_year_total_claims_value = st.number_input("Total Claims Value (Last 5 Years) ($)", min_value=0.0, value=0.0)
    license_points = st.number_input("License Points", min_value=0.0, max_value=30.0, value=0.0)
    licence_revoked = st.selectbox("Licence Revoked in Past", ["Yes", "No"])
    
if st.button("Predict Claim Risk", use_container_width=True):
    payload = {
        "num_young_drivers": num_young_drivers,
        "age": age,
        "num_of_children": num_of_children,
        "years_job_held_for": years_job_held_for,
        "income": income,
        "value_of_home": value_of_home,
        "commute_dist": commute_dist,
        "vehicle_value": vehicle_value,
        "policy_tenure": policy_tenure,
        "5_year_total_claims_value": five_year_total_claims_value,
        "5_year_num_of_claims": five_year_num_of_claims,
        "license_points": license_points,
        "vehicle_age": vehicle_age,
        "highest_education": highest_education,
        "single_parent": single_parent,
        "married": married,
        "gender": gender,
        "type_of_use": type_of_use,
        "licence_revoked": licence_revoked,
        "address_type": address_type,
        "occupation": occupation,
        "vehicle_type": vehicle_type
    }
    
    with st.spinner("Analyzing risk profile..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            st.divider()
            st.subheader("Prediction Results")
            
            prob = result["claim_probability"]
            decision = result["claim_decision"]
            est_value = result["estimated_claim_value"]
            
            rcol1, rcol2, rcol3 = st.columns(3)
            
            # Use Plotly Gauge for probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Claim Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d9f0a3"},
                        {'range': [30, 70], 'color': "#fecc5c"},
                        {'range': [70, 100], 'color': "#fd8d3c"}
                    ]
                }
            ))
            
            with rcol1:
                st.plotly_chart(fig, use_container_width=True)
                
            with rcol2:
                st.metric("Claim Expected?", "Yes ⚠️" if decision == 1 else "No ✅")
                if decision == 1:
                    st.error("High Risk Profile Detected")
                else:
                    st.success("Low Risk Profile Detected")
                    
            with rcol3:
                st.metric("Estimated Claim Cost", f"${est_value:,.2f}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API. Please ensure the FastAPI backend is running on {API_URL}.")
            st.code("uvicorn app.main:app --reload", language="bash")
