# ============================================================
# app.py — Churn Analysis Pro (Enhanced UI & Features)
# Real Data Upload • Personalized Retention • PDF Report
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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Page config
st.set_page_config(page_title="Churn Pro", page_icon="🚨", layout="wide")

# Custom CSS
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
st.markdown("<p style='text-align: center; font-size: 22px;'>AI-Powered Retention Intelligence</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your customer CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} customers")

        required = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints', 
                    'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

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

        with st.spinner("Training model..."):
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

        st.success("Model trained!")

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Churn Rate", f"{df['predicted_churn'].mean():.1%}", delta_color="inverse")
        revenue_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12
        col3.metric("Revenue at Risk (Annual)", f"₦{revenue_risk:,.0f}", delta_color="inverse")

        # All High-Risk Customers (not just top 10)
        high_risk = df[df['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)
        st.subheader(f"High-Risk Customers ({len(high_risk)} — All Shown)")
        st.dataframe(high_risk[['customer_id', 'name', 'monthly_spend_ngn', 'churn_probability']])

        # What-If
        st.subheader("What-If Retention Simulator")
        discount = st.slider("Discount %", 5, 50, 20)
        retention = st.slider("Retention Gain %", 10, 80, 40)
        
        saved = len(high_risk) * (retention / 100)
        saved_rev = saved * high_risk['monthly_spend_ngn'].mean() * 12
        cost = saved * high_risk['monthly_spend_ngn'].mean() * (discount / 100) * 12
        net = saved_rev - cost
        
        st.metric("Net Revenue Saved (Annual)", f"₦{net:,.0f}", delta_color="normal")

        # Personalized Recommendations (Unique for Each)
        st.subheader("Personalized Retention Plans")
        for _, row in high_risk.head(10).iterrows():  # Show for top 10, but all in PDF
            plan = f"Customer {row['customer_id']} ({row['name']}): High probability ({row['churn_probability']:.1%}). Recommend: {discount}% discount on next purchase to retain high-spend customer (₦{row['monthly_spend_ngn']:,.0f}/month)."
            st.info(plan)

        # PDF Report
        st.subheader("Generate PDF Report")
        if st.button("Download PDF Report"):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Churn Analysis Report", styles['Heading1']))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Date: {datetime.now().date()}", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Metrics table
            metrics_data = [
                ["Metric", "Value"],
                ["Total Customers", len(df)],
                ["Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}"],
                ["Annual Revenue at Risk", f"₦{revenue_risk:,.0f}"],
                ["Net Revenue Saved (What-If)", f"₦{net:,.0f}"]
            ]
            t = Table(metrics_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 14),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(t)
            elements.append(Spacer(1, 12))

            # High-risk table (sample)
            high_risk_data = [["ID", "Name", "Spend (₦)", "Probability"]] + high_risk.head(10).values.tolist()
            t2 = Table(high_risk_data)
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(Paragraph("Top High-Risk Customers", styles['Heading2']))
            elements.append(t2)

            doc.build(elements)
            buffer.seek(0)
            st.download_button("Download PDF Report", buffer, "churn_report.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your CSV to begin")

st.caption("Built by Freda Erinmwingbovo • Abuja, Nigeria")
