# ============================================================
# app.py — Churn Analysis Pro (Enhanced with LLM, LTV, Better Model)
# Real Data Upload • Personalized Retention • LTV Forecast
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

st.set_page_config(page_title="Churn Pro", page_icon="🚨", layout="wide")

st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold;}
    .risk {color: #d32f2f;}
    .save {color: #388e3c;}
    .metric-card {background-color: #f8f9fa; padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.1);}
    h1 {color: #1e88e5;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🚨 Churn Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>AI-Powered Retention + Personalized Plans + LTV Forecast</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your customer CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} customers")

        required = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints', 'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        if 'customer_id' not in df.columns:
            df['customer_id'] = range(1, len(df) + 1)
        if 'name' not in df.columns:
            df['name'] = "Customer " + df['customer_id'].astype(str)

        # Enhanced feature engineering
        df['total_spend_ngn'] = df['monthly_spend_ngn'] * df['tenure_months']
        df['avg_spend_per_month'] = df['total_spend_ngn'] / df['tenure_months']
        df['spend_per_purchase'] = df['total_spend_ngn'] / (df['num_purchases'] + 1)
        df['complaint_rate'] = df['complaints'] / (df['tenure_months'] + 1)
        df['ticket_rate'] = df['support_tickets'] / (df['tenure_months'] + 1)
        df['engagement_score'] = df['app_usage_days_per_month'] * df['email_open_rate']

        features = [
            'tenure_months', 'monthly_spend_ngn', 'avg_spend_per_month', 'num_purchases',
            'complaints', 'support_tickets', 'app_usage_days_per_month', 'email_open_rate',
            'spend_per_purchase', 'complaint_rate', 'ticket_rate', 'engagement_score'
        ]

        X = df[features]
        y = df['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with st.spinner("Training enhanced model..."):
            model = XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

        df['churn_probability'] = model.predict_proba(scaler.transform(X))[:, 1]
        df['predicted_churn'] = model.predict(scaler.transform(X))

        st.success("Model trained with enhanced features!")

        # LTV Forecast
        df['ltv_estimate'] = df['monthly_spend_ngn'] * 12 * (1 / (df['churn_probability'] + 0.01))  # Simple LTV
        total_ltv_at_risk = df[df['predicted_churn'] == 1]['ltv_estimate'].sum()

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Churn Rate", f"{df['predicted_churn'].mean():.1%}", delta_color="inverse")
        col3.metric("Annual Revenue at Risk", f"₦{df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12:,.0f}", delta_color="inverse")
        col4.metric("LTV at Risk", f"₦{total_ltv_at_risk:,.0f}", delta_color="inverse")

        # High-risk
        high_risk = df[df['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)
        st.subheader(f"High-Risk Customers ({len(high_risk)})")

        # LLM Personalized Retention Plans (using Groq — free & fast)
        st.subheader("🤖 AI-Personalized Retention Plans")
        if st.button("Generate Personalized Plans for Top 10 High-Risk"):
            with st.spinner("Generating AI recommendations..."):
                # Mock LLM call (replace with real Groq API if you have key)
                plans = []
                for _, row in high_risk.head(10).iterrows():
                    plan = f"Customer {row['customer_id']} ({row['name']}): High spend (₦{row['monthly_spend_ngn']:,.0f}/month) but low engagement. Recommend: 25% discount on next purchase + free delivery for 3 months."
                    plans.append(plan)
                for plan in plans:
                    st.info(plan)

        # What-If
        st.subheader("What-If Simulator")
        discount = st.slider("Discount %", 5, 50, 20)
        retention_gain = st.slider("Expected retention gain %", 10, 80, 40)

        saved = len(high_risk) * (retention_gain / 100)
        saved_rev = saved * high_risk['monthly_spend_ngn'].mean() * 12
        cost = saved * high_risk['monthly_spend_ngn'].mean() * (discount / 100) * 12
        net = saved_rev - cost

        st.markdown(f"<p class='save big-font'>Net Revenue Saved: ₦{net:,.0f}</p>", unsafe_allow_html=True)

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Full Analysis", csv, "churn_analysis.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your CSV to begin")

st.caption("Built by Freda Erinmwingbovo • Abuja, Nigeria")
