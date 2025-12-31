# ============================================================
# app.py — Churn Analysis Pro (Ultimate Version)
# AI-Powered Retention • SHAP Explainability • Revenue Impact • Personalized Plans • PDF & PPT Export
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# PDF Libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

# PPT (Optional)
try:
    from python_pptx import Presentation
    PPT_AVAILABLE = True
except ModuleNotFoundError:
    PPT_AVAILABLE = False

# ---------------- CONFIG & UI ----------------
st.set_page_config(page_title="Churn Analysis Pro", page_icon="🚨", layout="wide")

st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold;}
    .risk {color: #d32f2f;}
    .save {color: #388e3c;}
    .metric-card {background-color: #f8f9fa; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    h1 {text-align: center; color: #1e88e5;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚨 Churn Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>AI-Powered Retention Intelligence • Explainability • Actionable Insights</p>", unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload your customer CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df):,} customers")

        # Required columns
        required = [
            'tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints',
            'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn'
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Auto-generate customer identifiers if missing
        if 'customer_id' not in df.columns:
            df['customer_id'] = range(1, len(df) + 1)
        if 'name' not in df.columns:
            df['name'] = "Customer " + df['customer_id'].astype(str)

        # ---------------- FEATURE ENGINEERING ----------------
        df['total_spend_ngn'] = df['monthly_spend_ngn'] * df['tenure_months']
        df['spend_per_purchase'] = df['total_spend_ngn'] / (df['num_purchases'] + 1)
        df['complaint_rate'] = df['complaints'] / (df['tenure_months'] + 1)
        df['ticket_rate'] = df['support_tickets'] / (df['tenure_months'] + 1)
        df['engagement'] = df['app_usage_days_per_month'] * df['email_open_rate']

        features = [
            'tenure_months', 'monthly_spend_ngn', 'num_purchases',
            'complaints', 'support_tickets', 'app_usage_days_per_month',
            'email_open_rate', 'spend_per_purchase', 'complaint_rate',
            'ticket_rate', 'engagement'
        ]

        X = df[features]
        y = df['churn']

        # Train-test split & scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---------------- MODEL TRAINING ----------------
        with st.spinner("Training XGBoost model..."):
            model = XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

        # Predictions
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        df['churn_probability'] = model.predict_proba(scaler.transform(X))[:, 1]
        df['predicted_churn'] = (df['churn_probability'] > 0.5).astype(int)

        st.success("✅ Model trained successfully!")

        # ---------------- DASHBOARD METRICS ----------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}", delta_color="inverse")
        col3.metric("Model Accuracy", f"{acc:.2%}")
        col4.metric("ROC-AUC Score", f"{auc:.2f}")

        # Revenue at Risk
        annual_revenue_at_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12
        st.metric("Annual Revenue at Risk", f"₦{annual_revenue_at_risk:,.0f}", delta_color="inverse")

        # ---------------- HIGH-RISK CUSTOMERS ----------------
        high_risk_threshold = 0.7
        high_risk = df[df['churn_probability'] > high_risk_threshold].sort_values('churn_probability', ascending=False)

        st.subheader(f"🔴 High-Risk Customers ({len(high_risk)} total)")
        display_cols = ['customer_id', 'name', 'monthly_spend_ngn', 'tenure_months', 'churn_probability']
        st.dataframe(
            high_risk[display_cols].style.format({
                'churn_probability': '{:.1%}',
                'monthly_spend_ngn': '₦{:.0f}'
            }),
            use_container_width=True
        )

        # ---------------- WHAT-IF SIMULATOR ----------------
        st.subheader("💡 What-If Retention Simulator")
        col_a, col_b = st.columns(2)
        with col_a:
            discount_pct = st.slider("Discount Offered (%)", 5, 50, 20)
        with col_b:
            retention_gain_pct = st.slider("Expected Retention Gain (%)", 10, 80, 40)

        if len(high_risk) > 0:
            avg_monthly_spend = high_risk['monthly_spend_ngn'].mean()
            customers_saved = len(high_risk) * (retention_gain_pct / 100)
            revenue_saved_annual = customers_saved * avg_monthly_spend * 12
            campaign_cost_annual = customers_saved * avg_monthly_spend * (discount_pct / 100) * 12
            net_benefit = revenue_saved_annual - campaign_cost_annual

            c1, c2, c3 = st.columns(3)
            c1.metric("Customers Potentially Saved", f"{customers_saved:.0f}")
            c2.metric("Revenue Saved (Annual)", f"₦{revenue_saved_annual:,.0f}")
            c3.metric("Net Benefit After Cost", f"₦{net_benefit:,.0f}", delta_color="normal" if net_benefit > 0 else "inverse")
        else:
            st.info("No high-risk customers detected — great retention health!")

        # ---------------- PERSONALIZED RECOMMENDATIONS ----------------
        if len(high_risk) > 0:
            st.subheader("🎯 Personalized Retention Recommendations (Top 10)")
            for _, cust in high_risk.head(10).iterrows():
                rec = (
                    f"**{cust['name']} (ID: {cust['customer_id']})** — "
                    f"Churn Risk: **{cust['churn_probability']:.1%}** — "
                    f"Monthly Spend: ₦{cust['monthly_spend_ngn']:,.0f}\n\n"
                    f"→ Offer **{discount_pct}% discount** on next renewal or bundle.\n"
                    f"→ Send personalized re-engagement email campaign.\n"
                    f"→ Assign dedicated support rep due to high value."
                )
                st.info(rec)

        # ---------------- SHAP EXPLAINABILITY ----------------
        st.subheader("🧠 Model Explainability (SHAP Beeswarm)")
        if st.button("Generate Global SHAP Explanation"):
            with st.spinner("Computing SHAP values..."):
                explainer = shap.Explainer(model, X_train_scaled)
                shap_values = explainer(X_test_scaled[:500])  # Sample for speed
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.beeswarm(shap_values, show=False, max_display=15)
                st.pyplot(fig)
                plt.clf()

        # ---------------- PDF EXPORT ----------------
        st.subheader("📄 Export Executive Report")
        if st.button("Generate & Download PDF Report"):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Churn Analysis Pro – Executive Report", styles['Title']))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
            elements.append(Spacer(1, 20))

            # Metrics Table
            metrics_data = [
                ["Metric", "Value"],
                ["Total Customers", f"{len(df):,}"],
                ["Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}"],
                ["Annual Revenue at Risk", f"₦{annual_revenue_at_risk:,.0f}"],
                ["Model Accuracy", f"{acc:.2%}"],
                ["ROC-AUC Score", f"{auc:.2f}"]
            ]
            table = Table(metrics_data, colWidths=[300, 180])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))

            # High-risk preview
            elements.append(Paragraph("Top 10 High-Risk Customers", styles['Heading2']))
            risk_data = [["ID", "Name", "Spend ₦", "Risk"]] + [
                [row['customer_id'], row['name'], f"₦{row['monthly_spend_ngn']:,.0f}", f"{row['churn_probability']:.1%}"]
                for _, row in high_risk.head(10).iterrows()
            ]
            risk_table = Table(risk_data)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(risk_table)

            doc.build(elements)
            buffer.seek(0)
            st.download_button(
                "📥 Download PDF Report",
                buffer,
                file_name="churn_analysis_executive_report.pdf",
                mime="application/pdf"
            )

        # ---------------- PPT EXPORT (Optional) ----------------
        if PPT_AVAILABLE:
            if st.button("📊 Export Investor PowerPoint"):
                prs = Presentation()
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = "Churn Analysis Pro – Key Insights"
                content = slide.shapes.placeholders[1]
                text_frame = content.text_frame
                text_frame.text = (
                    f"Total Customers: {len(df):,}\n"
                    f"Predicted Churn Rate: {df['predicted_churn'].mean():.1%}\n"
                    f"Annual Revenue at Risk: ₦{annual_revenue_at_risk:,.0f}\n"
                    f"Model Accuracy: {acc:.2%} | AUC: {auc:.2f}\n"
                    f"High-Risk Customers: {len(high_risk)}"
                )
                ppt_buffer = io.BytesIO()
                prs.save(ppt_buffer)
                ppt_buffer.seek(0)
                st.download_button(
                    "📥 Download PowerPoint",
                    ppt_buffer,
                    file_name="churn_investor_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
        else:
            st.info("📊 PPT export unavailable in this deployment (requires local run).")

        # ---------------- CSV EXPORT ----------------
        st.download_button(
            "⬇️ Download Full Analysis CSV",
            df.to_csv(index=False).encode('utf-8'),
            "churn_analysis_full_with_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

else:
    st.info("👆 Please upload a CSV file with customer data to begin analysis.")

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • December 2025")
