# ============================================================
# app.py — Customer Churn Analysis Tool (Beautiful & Production-Like)
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

# Page config with dark mode support
st.set_page_config(
    page_title="Churn Analysis Pro",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for vibrant look
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .high-risk {color: #d32f2f; font-weight: bold;}
    .saved-revenue {color: #388e3c; font-weight: bold;}
    h1 {color: #1e88e5;}
    .stButton>button {background-color: #1e88e5; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("📉 Churn Analysis Pro")
st.markdown("**AI-Powered Retention Intelligence • Real Data • Real ₦ Impact**")

st.markdown("""
**Upload your customer data to uncover:**
- Who is at risk of churning
- How much revenue is in danger
- Smart retention strategies to save millions
""")

uploaded_file = st.file_uploader("📁 Upload your customer CSV", type="csv", help="Required columns: tenure_months, monthly_spend_ngn, num_purchases, complaints, support_tickets, app_usage_days_per_month, email_open_rate, churn")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} customers")

        # Required columns
        required = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints', 
                    'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}")
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

        with st.spinner("🤖 Training XGBoost model on your data..."):
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

        df['churn_probability'] = model.predict_proba(scaler.transform(X))[:, 1]
        df['predicted_churn'] = model.predict(scaler.transform(X))

        st.success("🎉 Model trained!")

        # Key metrics with color
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}", delta_color="inverse")
        revenue_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12
        col3.metric("Annual Revenue at Risk", f"₦{revenue_risk:,.0f}", delta_color="inverse")
        high_risk_count = len(df[df['churn_probability'] > 0.7])
        col4.metric("High-Risk Customers", f"{high_risk_count:,}", delta_color="inverse")

        # High-risk table
        high_risk = df[df['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)
        st.subheader(f"🔴 High-Risk Customers ({len(high_risk)})")
        st.dataframe(
            high_risk[['customer_id', 'name', 'monthly_spend_ngn', 'churn_probability']].head(20),
            use_container_width=True
        )

        # What-If
        st.subheader("💡 What-If Retention Scenario")
        col1, col2 = st.columns(2)
        with col1:
            discount = st.slider("Discount % for high-risk", 5, 50, 20)
        with col2:
            retention_improve = st.slider("Expected retention improvement %", 10, 80, 40)

        saved_customers = len(high_risk) * (retention_improve / 100)
        saved_revenue = saved_customers * high_risk['monthly_spend_ngn'].mean() * 12
        discount_cost = saved_customers * high_risk['monthly_spend_ngn'].mean() * (discount / 100) * 12
        net_save = saved_revenue - discount_cost

        st.metric("Estimated Annual Net Revenue Saved", f"₦{net_save:,.0f}", delta_color="normal")

        # Recommendations
        st.subheader("🚀 Automated Recommendations")
        st.info(f"Target {len(high_risk)} high-risk customers with {discount}% personalized discount")
        st.info("Focus on reducing complaints and improving app engagement")
        st.info("Prioritize high-spend customers for VIP retention programs")

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Full Predictions",
            csv,
            "churn_predictions.csv",
            "text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👆 Upload a CSV to begin analysis")
    st.markdown("**Required columns**: tenure_months, monthly_spend_ngn, num_purchases, complaints, support_tickets, app_usage_days_per_month, email_open_rate, churn")

st.caption("Built by Freda Erinmwingbovo • Production-Ready Churn Intelligence")
