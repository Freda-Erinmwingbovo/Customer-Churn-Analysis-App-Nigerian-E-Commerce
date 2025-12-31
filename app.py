# ============================================================
# app.py — Churn Analysis Pro (FLAGSHIP PORTFOLIO VERSION)
# AI-Powered Retention • Explainability • LTV • PDF • PPT
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from python_pptx import Presentation

import io

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Analysis Pro", page_icon="🚨", layout="wide")

st.markdown("""
<style>
.big-font {font-size: 46px; font-weight: bold;}
.metric-card {background:#f8f9fa; padding:20px; border-radius:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚨 Churn Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:22px;'>AI-Powered Retention + Explainability + LTV</p>", unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload your customer CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df):,} customers")

    required = [
        'tenure_months','monthly_spend_ngn','num_purchases','complaints',
        'support_tickets','app_usage_days_per_month','email_open_rate','churn'
    ]
    if not all(c in df.columns for c in required):
        st.error("Missing required columns.")
        st.stop()

    # ---------------- FEATURES ----------------
    df['total_spend'] = df['monthly_spend_ngn'] * df['tenure_months']
    df['spend_per_purchase'] = df['total_spend'] / (df['num_purchases'] + 1)
    df['complaint_rate'] = df['complaints'] / (df['tenure_months'] + 1)
    df['ticket_rate'] = df['support_tickets'] / (df['tenure_months'] + 1)
    df['engagement'] = df['app_usage_days_per_month'] * df['email_open_rate']

    features = [
        'tenure_months','monthly_spend_ngn','num_purchases',
        'complaints','support_tickets','app_usage_days_per_month',
        'email_open_rate','spend_per_purchase','complaint_rate',
        'ticket_rate','engagement'
    ]

    X = df[features]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---------------- MODEL ----------------
    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train_s, y_train)

    # ---------------- METRICS ----------------
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:,1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    df['churn_probability'] = model.predict_proba(scaler.transform(X))[:,1]
    df['predicted_churn'] = (df['churn_probability'] > 0.5).astype(int)

    # ---------------- DASHBOARD ----------------
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Predicted Churn", f"{df['predicted_churn'].mean():.1%}")
    c3.metric("Accuracy", f"{acc:.2%}")
    c4.metric("ROC-AUC", f"{auc:.2f}")

    # ---------------- SHAP ----------------
    st.subheader("🧠 Model Explainability (SHAP)")
    if st.button("Generate SHAP Explanation"):
        explainer = shap.Explainer(model, X_train_s)
        shap_values = explainer(X_test_s[:500])

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

    # ---------------- PDF EXPORT ----------------
    if st.button("📄 Export Executive PDF"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("<b>Churn Analysis Pro – Executive Summary</b>", styles['Title']),
            Paragraph(f"Customers analysed: {len(df):,}", styles['Normal']),
            Paragraph(f"Churn rate: {df['predicted_churn'].mean():.1%}", styles['Normal']),
            Paragraph(f"Model Accuracy: {acc:.2%}", styles['Normal']),
            Paragraph(f"ROC-AUC: {auc:.2f}", styles['Normal']),
        ]

        doc.build(content)
        buffer.seek(0)

        st.download_button(
            "Download PDF",
            buffer,
            file_name="churn_executive_summary.pdf",
            mime="application/pdf"
        )

    # ---------------- PPT EXPORT ----------------
    if st.button("📊 Export Investor PowerPoint"):
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Churn Analysis Pro"

        body = slide.shapes.placeholders[1].text = (
            f"Customers: {len(df):,}\n"
            f"Churn Rate: {df['predicted_churn'].mean():.1%}\n"
            f"Accuracy: {acc:.2%}\n"
            f"ROC-AUC: {auc:.2f}"
        )

        ppt_buffer = io.BytesIO()
        prs.save(ppt_buffer)
        ppt_buffer.seek(0)

        st.download_button(
            "Download PPT",
            ppt_buffer,
            file_name="churn_investor_deck.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

    # ---------------- CSV ----------------
    st.download_button(
        "⬇️ Download Full CSV Analysis",
        df.to_csv(index=False),
        "churn_analysis_full.csv",
        "text/csv"
    )

else:
    st.info("Upload your customer CSV to begin")

st.caption("Built by Freda Erinmwingbovo • Abuja, Nigeria")
