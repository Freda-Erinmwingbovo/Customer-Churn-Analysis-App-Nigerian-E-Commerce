# ============================================================
# Customer Churn Analysis App — Nigerian E-Commerce
# High-Accuracy XGBoost Model • Revenue Impact • What-If Scenarios
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("churn_model_xgboost.pkl")
    scaler = joblib.load("churn_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Title
st.title("🇳🇬 Customer Churn Analysis & Retention Tool")
st.markdown("**Predict churn, calculate revenue at risk, and simulate retention strategies**")

# Upload or use sample data
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload customer data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using sample data")
    # Use your generated df (or sample)
    # For demo, we'll use a small sample
    df = pd.read_csv("sample_churn_data.csv")  # You can generate and save this

# Feature engineering (same as training)
df['spend_per_purchase'] = df['total_spend_ngn'] / (df['num_purchases'] + 1)
df['recent_activity'] = df['app_usage_days_per_month'] * df['email_open_rate']
df['complaint_rate'] = df['complaints'] / (df['tenure_months'] + 1)

features = [
    'age', 'tenure_months', 'monthly_spend_ngn', 'num_purchases',
    'complaints', 'support_tickets', 'app_usage_days_per_month', 'email_open_rate',
    'spend_per_purchase', 'recent_activity', 'complaint_rate'
]

df_model = pd.get_dummies(df, columns=['gender'], drop_first=True)
X = df_model[features + ['gender_Male']]

X_scaled = scaler.transform(X)

# Predictions
df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
df['predicted_churn'] = model.predict(X_scaled)

# Dashboard
st.header("Churn Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{df['predicted_churn'].mean():.1%}")
col3.metric("Revenue at Risk", f"₦{df[df['predicted_churn'] == 1]['total_spend_ngn'].sum():,.0f}")

# High-risk customers
st.header("High-Risk Customers (Top 20)")
high_risk = df[df['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False).head(20)
st.dataframe(high_risk[['customer_id', 'name', 'monthly_spend_ngn', 'complaints', 'churn_probability']])

# What-If Scenario
st.header("What-If Retention Scenario")
discount = st.slider("Discount % for high-risk customers", 5, 50, 20)
high_risk_count = len(df[df['churn_probability'] > 0.7])
saved_revenue = high_risk_count * (discount / 100) * df[df['churn_probability'] > 0.7]['monthly_spend_ngn'].mean() * 12  # Annual
cost = high_risk_count * (discount / 100) * df[df['churn_probability'] > 0.7]['monthly_spend_ngn'].mean() * 12

st.write(f"Offering {discount}% discount to {high_risk_count} high-risk customers:")
st.write(f"Estimated annual revenue saved: ₦{saved_revenue:,.0f}")
st.write(f"Estimated cost of discount: ₦{cost:,.0f}")
st.write(f"Net gain: ₦{saved_revenue - cost:,.0f}")

# Recommendations
st.header("Automated Recommendations")
st.write("1. Contact high-risk customers with personalized offers")
st.write("2. Improve app experience for low-usage customers")
st.write("3. Reduce complaints through better support")

st.success("Churn analysis complete — take action to save revenue!")
