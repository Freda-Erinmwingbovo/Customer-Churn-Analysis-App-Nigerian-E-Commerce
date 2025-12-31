# ============================================================
# app.py — Churn Analysis Pro (Ultimate Version)
# AI-Powered Retention • SHAP Explainability • LTV • PDF • PPT • What-If
# Built by Freda Erinmwingbovo • Abuja, Nigeria • December 2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# Optional PPT export
try:
    from python_pptx import Presentation
    PPT_AVAILABLE = True
except ImportError:
    PPT_AVAILABLE = False

# ---------------- CONFIG & STYLE ----------------
st.set_page_config(page_title="Churn Analysis Pro", page_icon="🚨", layout="wide")

st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold;}
    .risk {color: #d32f2f;}
    .save {color: #388e3c;}
    .metric-card {background-color: #f8f9fa; padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.1);}
    h1 {color: #1e88e5; text-align: center;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚨 Churn Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>AI-Powered Retention • Explainability • LTV • Actionable Insights</p>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
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
            st.error(f"🚫 Missing required columns: {', '.join(missing)}")
            st.stop()

        # Optional identifiers
        if 'customer_id' not in df.columns:
            df['customer_id'] = range(1, len(df) + 1)
        if 'name' not in df.columns:
            df['name'] = "Customer " + df['customer_id'].astype(str)

        # ---------------- FEATURE ENGINEERING ----------------
        df['total_spend_ngn'] = df['monthly_spend_ngn'] * df['tenure_months']
        df['spend_per_purchase'] = df['total_spend_ngn'] / (df['num_purchases'] + 1)
        df['complaint_rate'] = df['complaints'] / (df['tenure_months'] + 1)
        df['ticket_rate'] = df['support_tickets'] / (df['tenure_months'] + 1)
        df['engagement_score'] = df['app_usage_days_per_month'] * df['email_open_rate']

        features = [
            'tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints',
            'support_tickets', 'app_usage_days_per_month', 'email_open_rate',
            'spend_per_purchase', 'complaint_rate', 'ticket_rate', 'engagement_score'
        ]

        X = df[features]
        y = df['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)

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

        df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
        df['predicted_churn'] = (df['churn_probability'] > 0.5).astype(int)

        # LTV Estimate (simple but effective)
        retention_factor = 1 / (df['churn_probability'] + 0.01)
        df['ltv_estimate'] = df['monthly_spend_ngn'] * 12 * retention_factor

        total_revenue_at_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12
        total_ltv_at_risk = df[df['predicted_churn'] == 1]['ltv_estimate'].sum()

        # ---------------- DASHBOARD METRICS ----------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}", delta_color="inverse")
        col3.metric("Annual Revenue at Risk", f"₦{total_revenue_at_risk:,.0f}", delta_color="inverse")
        col4.metric("LTV at Risk", f"₦{total_ltv_at_risk:,.0f}", delta_color="inverse")

        col5, col6 = st.columns(2)
        col5.metric("Model Accuracy", f"{acc:.1%}")
        col6.metric("ROC-AUC Score", f"{auc:.3f}")

        st.markdown("---")

        # ---------------- TABS FOR ORGANIZATION ----------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔴 High-Risk Customers", "🧠 Model Explainability", "💰 What-If Simulator", "📄 Export Reports", "📊 Full Data"
        ])

        # ---------------- TAB 1: High-Risk Customers ----------------
        with tab1:
            st.subheader("Top Customers at Risk of Churning")
            high_risk = df[df['churn_probability'] > 0.6].sort_values('churn_probability', ascending=False).copy()
            high_risk_display = high_risk[['customer_id', 'name', 'churn_probability', 'monthly_spend_ngn', 'tenure_months', 'engagement_score', 'ltv_estimate']].round(2)
            high_risk_display['churn_probability'] = (high_risk_display['churn_probability'] * 100).round(1).astype(str) + "%"
            high_risk_display['monthly_spend_ngn'] = high_risk_display['monthly_spend_ngn'].apply(lambda x: f"₦{x:,.0f}")
            high_risk_display['ltv_estimate'] = high_risk_display['ltv_estimate'].apply(lambda x: f"₦{x:,.0f}")

            st.dataframe(high_risk_display.head(50), use_container_width=True)

            if not high_risk.empty:
                for _, cust in high_risk.head(10).iterrows():
                    with st.expander(f"🔴 {cust['name']} (ID: {cust['customer_id']}) – {cust['churn_probability']:.1%} risk"):
                        st.write(f"**Monthly Spend:** ₦{cust['monthly_spend_ngn']:,.0f} | **Tenure:** {cust['tenure_months']} months")
                        st.write(f"**LTV Estimate:** ₦{cust['ltv_estimate']:,.0f}")
                        st.write("**Suggested Action:** Offer personalized discount, loyalty bonus, or proactive support call.")

        # ---------------- TAB 2: SHAP Explainability ----------------
        with tab2:
            st.subheader("Global Model Explainability (SHAP Beeswarm)")
            if st.button("Generate SHAP Beeswarm Plot"):
                with st.spinner("Computing SHAP values..."):
                    explainer = shap.Explainer(model, X_train_scaled)
                    shap_values = explainer(X_test_scaled[:500])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.beeswarm(shap_values, show=False, max_display=12)
                    st.pyplot(fig)
                    plt.clf()

        # ---------------- TAB 3: What-If Simulator ----------------
        with tab3:
            st.subheader("Retention Campaign Simulator")
            col1, col2 = st.columns(2)
            with col1:
                discount_pct = st.slider("Discount Offered (%)", 5, 50, 20)
            with col2:
                retention_gain_pct = st.slider("Expected Retention Improvement (%)", 10, 80, 40)

            target_customers = len(high_risk)
            saved_customers = target_customers * (retention_gain_pct / 100)
            avg_monthly = high_risk['monthly_spend_ngn'].mean() if not high_risk.empty else 0

            revenue_saved = saved_customers * avg_monthly * 12
            campaign_cost = saved_customers * avg_monthly * (discount_pct / 100) * 12
            net_benefit = revenue_saved - campaign_cost

            st.markdown(f"<p class='save big-font'>Projected Net Revenue Saved: ₦{net_benefit:,.0f}</p>", unsafe_allow_html=True)
            st.write(f"**Customers Retained:** ~{saved_customers:.0f} out of {target_customers}")
            st.write(f"**Revenue Saved:** ₦{revenue_saved:,.0f} | **Campaign Cost:** ₦{campaign_cost:,.0f}")

        # ---------------- TAB 4: Export Reports ----------------
        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Generate Executive PDF Report"):
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4)
                    styles = getSampleStyleSheet()
                    story = []

                    story.append(Paragraph("Churn Analysis Pro – Executive Report", styles['Title']))
                    story.append(Spacer(1, 12))
                    story.append(Paragraph(f"Analysis Date: December 2025", styles['Normal']))
                    story.append(Paragraph(f"Total Customers: {len(df):,}", styles['Normal']))
                    story.append(Paragraph(f"Predicted Churn Rate: {df['predicted_churn'].mean():.1%}", styles['Normal']))
                    story.append(Paragraph(f"Annual Revenue at Risk: ₦{total_revenue_at_risk:,.0f}", styles['Normal']))
                    story.append(Paragraph(f"LTV at Risk: ₦{total_ltv_at_risk:,.0f}", styles['Normal']))
                    story.append(Paragraph(f"Model Accuracy: {acc:.1%} | AUC: {auc:.3f}", styles['Normal']))

                    buffer.seek(0)
                    st.download_button(
                        "Download PDF Report",
                        buffer,
                        "churn_executive_report.pdf",
                        mime="application/pdf"
                    )

            with col2:
                if PPT_AVAILABLE:
                    if st.button("📊 Generate Investor PowerPoint"):
                        prs = Presentation()
                        slide = prs.slides.add_slide(prs.slide_layouts[1])
                        slide.shapes.title.text = "Churn Analysis Pro – Key Insights"
                        content = slide.shapes.placeholders[1].text_frame
                        content.add_paragraph().text = f"Total Customers: {len(df):,}"
                        content.add_paragraph().text = f"Predicted Churn Rate: {df['predicted_churn'].mean():.1%}"
                        content.add_paragraph().text = f"Annual Revenue at Risk: ₦{total_revenue_at_risk:,.0f}"
                        content.add_paragraph().text = f"LTV at Risk: ₦{total_ltv_at_risk:,.0f}"
                        content.add_paragraph().text = f"Model Performance: Accuracy {acc:.1%}, AUC {auc:.3f}"

                        ppt_buffer = io.BytesIO()
                        prs.save(ppt_buffer)
                        ppt_buffer.seek(0)
                        st.download_button(
                            "Download PowerPoint",
                            ppt_buffer,
                            "churn_investor_presentation.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        )
                else:
                    st.info("📊 PPT export unavailable in this environment. Run locally to enable.")

        # ---------------- TAB 5: Full Data ----------------
        with tab5:
            st.subheader("Download Enriched Dataset")
            st.download_button(
                label="⬇️ Download Full Analysis CSV (with predictions & LTV)",
                data=df.to_csv(index=False).encode(),
                file_name="churn_analysis_enriched.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👆 Please upload a CSV file containing customer data to begin analysis.")

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • December 2025")
