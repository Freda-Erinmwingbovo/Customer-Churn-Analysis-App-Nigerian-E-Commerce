# ============================================================
# app.py — Churn Analysis Pro (Vibrant & Professional UI)
# Real Data Upload • Trains on Your CSV • Shows ₦ Impact
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

# Page config
st.set_page_config(page_title="Churn Pro", page_icon="🚨", layout="wide")

# Custom CSS for excitement
st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold; color: #d32f2f;}
    .metric-card {background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.15); text-align: center;}
    .risk {color: #d32f2f;}
    .save {color: #388e3c;}
    .header {color: #1e88e5; font-size: 28px; font-weight: bold;}
    .stButton>button {background-color: #1e88e5; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #1e88e5;'>🚨 Churn Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>AI-Powered Retention Intelligence • Save Millions in Revenue</p>", unsafe_allow_html=True)

st.markdown("### 📁 Upload Your Customer Data to Unlock Insights")

uploaded_file = st.file_uploader("Choose CSV file", type="csv", label_visibility="collapsed")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} customers — Ready for analysis!")

        # Required columns check
        required = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints', 
                    'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Optional
        if 'customer_id' not in df.columns:
            df['customer_id'] = range(1, len(df) + 1)
        if 'name' not in df.columns:
            df['name'] = "Customer " + df['customer_id'].astype(str)

        # Feature engineering
        df['total_spend_ngn'] = df['monthly_spend_ngn'] * df['tenure_months']
        df['spend_per_purchase'] = df['total_spend_ngn'] / (df['num_purchases'] + 1)
        df['recent_activity'] = df['app_usage_days_per_month'] * df['email_open_rate']
        df['complaint_rate'] = df['complaints'] / (df['tenure_months'] + 1)

        features = [
            'tenure_months', 'monthly_spend_ngn', 'num_purchases',
            'complaints', 'support_tickets', 'app_usage_days_per_month', 'email_open_rate',
            'spend_per_purchase', 'recent_activity', 'complaint_rate'
        ]

        X = df[features]
        y = df['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with st.spinner("🤖 Training powerful model..."):
            model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train_scaled, y_train)

        df['churn_probability'] = model.predict_proba(scaler.transform(X))[:, 1]
        df['predicted_churn'] = model.predict(scaler.transform(X))

        st.balloons()
        st.success("🎉 Model trained — Insights ready!")

        # Vibrant metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label="Total Customers", value=f"{len(df):,}")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label="Churn Rate", value=f"{df['predicted_churn'].mean():.1%}", delta_color="inverse")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            revenue_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label="Revenue at Risk (Annual)", value=f"₦{revenue_risk:,.0f}", delta_color="inverse")
            st.markdown("</div>", unsafe_allow_html=True)
        with col4:
            high_risk = len(df[df['churn_probability'] > 0.7])
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label="High-Risk Customers", value=f"{high_risk:,}", delta_color="inverse")
            st.markdown("</div>", unsafe_allow_html=True)

        # High-risk table
        st.markdown("<p class='header'>🔴 High-Risk Customers</p>", unsafe_allow_html=True)
        high_risk_df = df[df['churn_probability'] > 0.7][['customer_id', 'name', 'monthly_spend_ngn', 'churn_probability']].sort_values('churn_probability', ascending=False)
        st.dataframe(high_risk_df.head(20), use_container_width=True)

        # What-If
        st.markdown("<p class='header'>💡 What-If Retention Simulator</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            discount = st.slider("Discount for high-risk (%)", 5, 50, 20)
        with col2:
            retention = st.slider("Expected retention gain (%)", 10, 80, 40)

        saved = len(high_risk_df) * (retention / 100)
        saved_rev = saved * high_risk_df['monthly_spend_ngn'].mean() * 12
        cost = saved * high_risk_df['monthly_spend_ngn'].mean() * (discount / 100) * 12
        net = saved_rev - cost

        st.markdown(f"<p class='save big-font'>Net Revenue Saved: ₦{net:,.0f}</p>", unsafe_allow_html=True)

        # Recommendations
        st.markdown("<p class='header'>🚀 Smart Recommendations</p>", unsafe_allow_html=True)
        st.success(f"Target {len(high_risk_df)} high-risk customers with {discount}% personalized offer")
        st.success("Reduce complaints through proactive support")
        st.success("Boost app engagement with push notifications")

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Full Report", csv, "churn_analysis_report.csv", "text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your customer CSV to unlock powerful churn insights")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria</p>", unsafe_allow_html=True)
