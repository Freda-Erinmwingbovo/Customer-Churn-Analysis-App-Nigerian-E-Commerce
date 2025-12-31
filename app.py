# ============================================================
# Customer Churn Analysis App — Nigerian E-Commerce
# High-Accuracy XGBoost Model • Revenue Impact • What-If Scenarios
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

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

# Upload data — REQUIRED
st.sidebar.header("Upload Your Customer Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (required columns: age, tenure_months, monthly_spend_ngn, num_purchases, complaints, support_tickets, app_usage_days_per_month, email_open_rate, gender, total_spend_ngn)",
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df):,} customers successfully")
else:
    st.info("👆 Please upload your customer CSV file to begin the analysis")
    st.stop()  # App stops until file is uploaded

# Feature engineering
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
col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}")
col3.metric("Revenue at Risk", f"₦{df[df['predicted_churn'] == 1]['total_spend_ngn'].sum():,.0f}")

# High-risk customers
st.header("High-Risk Customers (Churn Probability > 70%)")
high_risk = df[df['churn_probability'] > 0.7][['customer_id', 'name', 'state', 'monthly_spend_ngn', 'complaints', 'app_usage_days_per_month', 'churn_probability']]
high_risk = high_risk.sort_values('churn_probability', ascending=False)
st.dataframe(high_risk)

# What-If Scenario
st.header("Retention Scenario Simulator")
discount = st.slider("Offer % discount to high-risk customers", 5, 50, 20)
high_risk_count = len(high_risk)
annual_saved = high_risk_count * high_risk['monthly_spend_ngn'].mean() * 12 * (1 - discount/100)
cost = high_risk_count * high_risk['monthly_spend_ngn'].mean() * 12 * (discount/100)

st.write(f"Offering {discount}% discount to {high_risk_count:,} high-risk customers:")
st.write(f"Estimated annual revenue retained: ₦{annual_saved:,.0f}")
st.write(f"Cost of discount: ₦{cost:,.0f}")
st.write(f"**Net gain: ₦{annual_saved - cost:,.0f}**")

# Recommendations
st.header("Automated Retention Recommendations")
st.write("- Prioritize outreach to customers with >1 complaint and low app usage")
st.write("- Offer loyalty rewards to long-tenure, high-spend customers showing decline")
st.write("- Improve onboarding for new customers (tenure < 6 months)")
st.write("- Segment marketing: target low-engagement users with re-activation campaigns")

st.success("Analysis complete — take action to protect your revenue!")

# Footer
st.markdown("---")
st.markdown("Built by **Freda Erinmwingbovo** • Abuja, Nigeria • December 2025")
