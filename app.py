# ============================================================
# app.py — Customer Churn Analysis (Production-Like with Real Data Upload)
# Trains on User's CSV • Predicts Churn • Shows ₦ Impact
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Analysis", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Analysis Tool")
st.markdown("**Production-Like • Real Data Upload • Trains on Your Customers**")

st.markdown("""
Upload your customer data (CSV) to:
- Predict who will churn
- See revenue at risk (in ₦)
- Simulate retention strategies
- Get automated recommendations
""")

# File upload
uploaded_file = st.file_uploader("Upload your customer CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Data loaded: {len(df)} customers")

        # Required columns
        required_cols = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints', 'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Feature engineering
        df['spend_per_purchase'] = df['total_spend_ngn'] / (df['num_purchases'] + 1) if 'total_spend_ngn' in df else df['monthly_spend_ngn'] * df['tenure_months'] / (df['num_purchases'] + 1)
        df['recent_activity'] = df['app_usage_days_per_month'] * df['email_open_rate']
        df['complaint_rate'] = df['complaints'] / (df['tenure_months'] + 1)

        features = [
            'age' if 'age' in df else np.random.normal(32, 10, len(df)),  # fallback
            'tenure_months', 'monthly_spend_ngn', 'num_purchases',
            'complaints', 'support_tickets', 'app_usage_days_per_month', 'email_open_rate',
            'spend_per_purchase', 'recent_activity', 'complaint_rate'
        ]

        # Handle optional age
        if 'age' not in df.columns:
            df['age'] = np.random.normal(32, 10, len(df))
            df['age'] = df['age'].clip(18, 70).astype(int)

        X = df[features]
        y = df['churn']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
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

        # Results
        st.success("Model trained!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}")
        revenue_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12  # Annual
        col3.metric("Annual Revenue at Risk (₦)", f"{revenue_risk:,.0f}")

        # High-risk customers
        high_risk = df[df['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)
        st.subheader("High-Risk Customers (Probability >70%)")
        st.dataframe(high_risk[['customer_id', 'name', 'monthly_spend_ngn', 'churn_probability']].head(20))

        # What-if
        st.subheader("What-If Retention Scenario")
        discount_rate = st.slider("Discount for high-risk customers (%)", 5, 50, 20)
        retention_rate = st.slider("Expected retention improvement (%)", 10, 80, 40)

        saved_customers = len(high_risk) * (retention_rate / 100)
        saved_revenue = saved_customers * high_risk['monthly_spend_ngn'].mean() * 12
        cost = saved_customers * high_risk['monthly_spend_ngn'].mean() * (discount_rate / 100) * 12

        net_save = saved_revenue - cost

        st.metric("Estimated Annual Net Revenue Saved (₦)", f"{net_save:,.0f}")

        # Recommendations
        st.subheader("Automated Recommendations")
        st.write("- Contact high-risk customers with personalized offers")
        st.write(f"- Target {len(high_risk)} customers with {discount_rate}% discount")
        st.write("- Focus on reducing complaints and improving app usage")

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV with columns: tenure_months, monthly_spend_ngn, num_purchases, complaints, support_tickets, app_usage_days_per_month, email_open_rate, churn (and optional: customer_id, name, age, gender)")

st.caption("Built by Freda Erinmwingbovo • Production-Like Churn Tool")
