import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Load trained model and feature columns
model, feature_columns = pickle.load(open('model.pkl', 'rb'))

st.title("📊 Customer Churn Prediction & Analytics")
st.write("This machine learning application predicts whether a customer will leave the service and calculates the exact churn risk percentage.")

# Sidebar for User Inputs
st.sidebar.header("🔍 Customer Details Input")

gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 12)
monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=800.0)
contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Prediction Button on Sidebar
if st.sidebar.button('Predict Churn Risk', type="primary"):
    # Mapping inputs to numerical values
    gender_val = 1 if gender == 'Male' else 0
    partner_val = 1 if partner == 'Yes' else 0
    dependents_val = 1 if dependents == 'Yes' else 0
    
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract_val = contract_map[contract]
    
    payment_map = {'Electronic check': 2, 'Mailed check': 3, 'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1}
    payment_val = payment_map[payment_method]

    # Creating input array
    features_list = [senior_citizen, tenure, monthly_charges, total_charges, gender_val, partner_val, dependents_val, contract_val, payment_val]
    prediction = model.predict([features_list + [0]*(len(feature_columns)-len(features_list))])
    prediction_proba = model.predict_proba([features_list + [0]*(len(feature_columns)-len(features_list))])

    churn_prob = prediction_proba[0][1] * 100

    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"⚠️ This customer is likely to **Churn** (High Risk).")
        else:
            st.success(f"✅ This customer is likely to **Stay** (Loyal Customer).")
            
    with col2:
        st.subheader("Churn Probability Score")
        st.metric(label="Risk Percentage", value=f"{churn_prob:.2f}%")