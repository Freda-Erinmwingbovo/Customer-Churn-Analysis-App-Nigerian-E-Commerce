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
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Churn Analysis - Nigerian E-Commerce",
    page_icon="📉",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("churn_model_xgboost.pkl")
    scaler = joblib.load("churn_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Title
st.title("📉 Customer Churn Analysis & Retention Tool")
st.markdown("**AI-powered churn prediction for Nigerian e-commerce** | High-accuracy model with revenue impact insights")

# Sample data (for demo — replace with your full df if needed)
if 'df' not in st.session_state:
    # Load or generate sample data
    st.session_state.df = pd.read_csv("sample_churn_data.csv")  # You can generate and save this

df = st.session_state.df.copy()

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
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}")
col3.metric("High-Risk Customers", len(df[df['churn_probability'] > 0.7]))
col4.metric("Revenue at Risk", f"₦{df[df['predicted_churn'] == 1]['total_spend_ngn'].sum():,.0f}")

# Churn distribution
fig = px.histogram(df, x='churn_probability', nbins=50, title="Distribution of Churn Probability")
st.plotly_chart(fig, use_container_width=True)

# High-risk customers table
st.subheader("High-Risk Customers (Probability > 70%)")
high_risk = df[df['churn_probability'] > 0.7][['customer_id', 'name', 'monthly_spend_ngn', 'complaints', 'app_usage_days_per_month', 'churn_probability']]
high_risk['churn_probability'] = (high_risk['churn_probability'] * 100).round(1).astype(str) + "%"
st.dataframe(high_risk.sort_values('churn_probability', ascending=False), use_container_width=True)

# What-If Scenario
st.subheader("What-If Retention Scenario")
discount = st.slider("Offer discount to high-risk customers (%)", 5, 50, 20)
high_risk_count = len(df[df['churn_probability'] > 0.7])
avg_monthly_spend = df[df['churn_probability'] > 0.7]['monthly_spend_ngn'].mean()

annual_saved = high_risk_count * avg_monthly_spend * 12 * (1 - discount/100)
cost = high_risk_count * avg_monthly_spend * 12 * (discount/100)
net_gain = annual_saved - cost

st.write(f"**Scenario**: Offer {discount}% discount to {high_risk_count} high-risk customers")
st.write(f"Estimated annual revenue retained: **₦{annual_saved:,.0f}**")
st.write(f"Cost of discount: **₦{cost:,.0f}**")
st.write(f"**Net gain**: **₦{net_gain:,.0f}**")

# Recommendations
st.subheader("Automated Retention Recommendations")
st.write("1. Contact high-risk customers with personalized offers (e.g., free shipping, loyalty points)")
st.write("2. Improve app experience for low-usage customers")
st.write("3. Address complaints quickly to prevent escalation")
st.write("4. Run targeted email campaigns for low open-rate customers")

st.success("Churn analysis complete — take action to save revenue!")

# Footer
st.markdown("---")
st.markdown("Built by **Freda Erinmwingbovo** • Abuja, Nigeria • December 2025")
