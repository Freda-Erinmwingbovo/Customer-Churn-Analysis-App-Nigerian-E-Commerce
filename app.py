# ============================================================
# app.py — Customer Churn Analysis Tool (Production-Like)
# Real Data Upload • Trains on Your CSV • Shows ₦ Impact
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Analysis", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Analysis Tool")
st.markdown("**Production-Like • Upload Your Real Customer Data**")

st.markdown("""
Upload your CSV to:
- Predict churn risk
- See revenue at risk in ₦
- Simulate retention strategies
- Get automated recommendations
""")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} customers")

        # Required columns
        required = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints', 'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Optional columns
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

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost
        with st.spinner("Training model on your data..."):
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

        # Predictions
        df['churn_probability'] = model.predict_proba(scaler.transform(X))[:, 1]
        df['predicted_churn'] = model.predict(scaler.transform(X))

        st.success("Model trained!")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}")
        revenue_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12
        col3.metric("Annual Revenue at Risk (₦)", f"{revenue_risk:,.0f}")

        # High-risk customers
        high_risk = df[df['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)
        st.subheader(f"High-Risk Customers ({len(high_risk)})")
        st.dataframe(high_risk[['customer_id', 'name', 'monthly_spend_ngn', 'churn_probability']])

        # What-If Scenario
        st.subheader("What-If Retention Scenario")
        discount = st.slider("Discount % for high-risk customers", 5, 50, 20)
        retention_improve = st.slider("Expected retention improvement %", 10, 80, 40)

        saved_customers = len(high_risk) * (retention_improve / 100)
        saved_revenue = saved_customers * high_risk['monthly_spend_ngn'].mean() * 12
        discount_cost = saved_customers * high_risk['monthly_spend_ngn'].mean() * (discount / 100) * 12
        net_save = saved_revenue - discount_cost

        st.metric("Estimated Annual Net Revenue Saved (₦)", f"{net_save:,.0f}")

        # Recommendations
        st.subheader("Automated Recommendations")
        st.write(f"- Target {len(high_risk)} high-risk customers with {discount}% discount")
        st.write("- Focus on reducing complaints and improving app usage")
        st.write("- Prioritize customers with high monthly spend")

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Predictions CSV", csv, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.info("Upload a CSV with columns: tenure_months, monthly_spend_ngn, num_purchases, complaints, support_tickets, app_usage_days_per_month, email_open_rate, churn (optional: customer_id, name)")

st.caption("Built by Freda Erinmwingbovo • Production-Like Churn Tool")
