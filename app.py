# ============================================================
# app.py — Churn Analysis Pro (Fully Fixed & Enhanced)
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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

# Optional PPT
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
    h1 {color: #1e88e5; text-align: center;}
    .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚨 Churn Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>AI-Powered Retention • Explainability • LTV • Actionable Insights</p>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload your customer CSV (must include required columns)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df):,} customers")

        required = [
            'tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints',
            'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn'
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"🚫 Missing required columns: {', '.join(missing)}")
            st.stop()

        if 'customer_id' not in df.columns:
            df['customer_id'] = range(1, len(df) + 1)
        if 'name' not in df.columns:
            df['name'] = "Customer " + df['customer_id'].astype(str)

        # Feature Engineering
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

        with st.spinner("Training XGBoost model..."):
            model = XGBClassifier(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42
            )
            model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
        df['predicted_churn'] = (df['churn_probability'] > 0.5).astype(int)

        # LTV Estimate
        df['ltv_estimate'] = df['monthly_spend_ngn'] * 12 * (1 / (df['churn_probability'] + 0.01))

        total_revenue_at_risk = df[df['predicted_churn'] == 1]['monthly_spend_ngn'].sum() * 12
        total_ltv_at_risk = df[df['predicted_churn'] == 1]['ltv_estimate'].sum()

        # Dashboard Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}", delta_color="inverse")
        col3.metric("Annual Revenue at Risk", f"₦{total_revenue_at_risk:,.0f}", delta_color="inverse")
        col4.metric("LTV at Risk", f"₦{total_ltv_at_risk:,.0f}", delta_color="inverse")

        col5, col6 = st.columns(2)
        col5.metric("Model Accuracy", f"{acc:.1%}")
        col6.metric("ROC-AUC", f"{auc:.3f}")

        st.markdown("---")

        # ---------------- TABS (NOW ALWAYS VISIBLE) ----------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔴 High-Risk Customers", "🧠 Model Explainability", "💰 What-If Simulator", "📄 Export Reports", "📊 Full Data"
        ])

        # TAB 1: High-Risk Customers
        with tab1:
            st.subheader("Customers at Risk of Churning")
            risk_threshold = st.slider("Risk Threshold for Display", 0.3, 0.9, 0.6, 0.05)
            high_risk = df[df['churn_probability'] > risk_threshold].sort_values('churn_probability', ascending=False).copy()

            if high_risk.empty:
                st.info(f"No customers above {risk_threshold:.0%} risk threshold. Great retention health! 🎉")
            else:
                display_cols = ['customer_id', 'name', 'churn_probability', 'monthly_spend_ngn', 'tenure_months', 'engagement_score', 'ltv_estimate']
                display_df = high_risk[display_cols].copy()
                display_df['churn_probability'] = (display_df['churn_probability'] * 100).round(1).astype(str) + "%"
                display_df['monthly_spend_ngn'] = display_df['monthly_spend_ngn'].apply(lambda x: f"₦{x:,.0f}")
                display_df['ltv_estimate'] = display_df['ltv_estimate'].apply(lambda x: f"₦{x:,.0f}")

                st.dataframe(display_df.head(100), use_container_width=True)

                st.write(f"**{len(high_risk)} customers** at >{risk_threshold:.0%} risk")

                for _, cust in high_risk.head(10).iterrows():
                    with st.expander(f"🔴 {cust['name']} (ID: {cust['customer_id']}) – Risk: {cust['churn_probability']:.1%}"):
                        st.write(f"**Monthly Spend:** ₦{cust['monthly_spend_ngn']:,.0f}")
                        st.write(f"**Tenure:** {cust['tenure_months']} months")
                        st.write(f"**Engagement Score:** {cust['engagement_score']:.2f}")
                        st.write(f"**Estimated LTV:** ₦{cust['ltv_estimate']:,.0f}")
                        st.write("**Recommended Action:** Personalized discount, loyalty reward, or proactive outreach.")

        # TAB 2: SHAP Explainability
        with tab2:
            st.subheader("Global Model Explainability")
            if st.button("Generate SHAP Beeswarm Plot"):
                with st.spinner("Computing SHAP values (may take 10-20 seconds)..."):
                    explainer = shap.Explainer(model, X_train_scaled)
                    shap_values = explainer(X_test_scaled[:300])
                    fig, ax = plt.subplots(figsize=(10, 7))
                    shap.plots.beeswarm(shap_values, max_display=12, show=False)
                    st.pyplot(fig)
                    plt.clf()

        # TAB 3: What-If Simulator (NOW FULLY WORKING)
        with tab3:
            st.subheader("Retention Campaign Simulator")
            if high_risk.empty:
                st.info("No high-risk customers to simulate campaigns on yet. Adjust threshold in Tab 1.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    discount_pct = st.slider("Discount Offered (%)", 5, 60, 25, key="discount")
                with col2:
                    retention_gain_pct = st.slider("Expected Retention Gain (%)", 10, 90, 50, key="retention")

                target_count = len(high_risk)
                expected_retained = target_count * (retention_gain_pct / 100)
                avg_monthly_spend = high_risk['monthly_spend_ngn'].mean()

                revenue_saved = expected_retained * avg_monthly_spend * 12
                campaign_cost = expected_retained * avg_monthly_spend * (discount_pct / 100) * 12
                net_benefit = revenue_saved - campaign_cost

                st.markdown(f"<p class='save big-font'>Projected Net Revenue Saved: ₦{net_benefit:,.0f}</p>", unsafe_allow_html=True)
                st.write(f"**Target Customers:** {target_count}")
                st.write(f"**Expected Retained:** ~{expected_retained:.0f}")
                st.write(f"**Gross Revenue Saved:** ₦{revenue_saved:,.0f}")
                st.write(f"**Campaign Cost:** ₦{campaign_cost:,.0f}")

        # TAB 4: Export Reports (ENHANCED PDF WITH HIGH-RISK TABLE)
        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Generate Detailed Executive PDF"):
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
                    styles = getSampleStyleSheet()
                    story = []

                    story.append(Paragraph("Churn Analysis Pro – Executive Report", styles['Title']))
                    story.append(Spacer(1, 20))
                    story.append(Paragraph("Prepared by: Freda Erinmwingbovo", styles['Normal']))
                    story.append(Paragraph(f"Date: December 31, 2025", styles['Normal']))
                    story.append(Spacer(1, 30))

                    # Key Metrics Table
                    metrics_data = [
                        ["Metric", "Value"],
                        ["Total Customers", f"{len(df):,}"],
                        ["Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}"],
                        ["Annual Revenue at Risk", f"₦{total_revenue_at_risk:,.0f}"],
                        ["LTV at Risk", f"₦{total_ltv_at_risk:,.0f}"],
                        ["Model Accuracy", f"{acc:.1%}"],
                        ["ROC-AUC Score", f"{auc:.3f}"]
                    ]
                    metrics_table = Table(metrics_data, colWidths=[3*inch, 2.5*inch])
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e88e5")),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,0), 14),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                        ('GRID', (0,0), (-1,-1), 1, colors.black)
                    ]))
                    story.append(metrics_table)
                    story.append(Spacer(1, 30))

                    # Top High-Risk Customers
                    story.append(Paragraph("Top 20 High-Risk Customers", styles['Heading2']))
                    top_risk = high_risk.head(20)[['name', 'churn_probability', 'monthly_spend_ngn', 'ltv_estimate']]
                    top_risk['churn_probability'] = top_risk['churn_probability'].apply(lambda x: f"{x:.1%}")
                    top_risk['monthly_spend_ngn'] = top_risk['monthly_spend_ngn'].apply(lambda x: f"₦{x:,.0f}")
                    top_risk['ltv_estimate'] = top_risk['ltv_estimate'].apply(lambda x: f"₦{x:,.0f}")

                    risk_data = [["Name", "Churn Risk", "Monthly Spend", "LTV"]]
                    for _, row in top_risk.iterrows():
                        risk_data.append([row['name'], row['churn_probability'], row['monthly_spend_ngn'], row['ltv_estimate']])

                    risk_table = Table(risk_data, colWidths=[2*inch, 1.2*inch, 1.3*inch, 1.5*inch])
                    risk_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                        ('FONTSIZE', (0,0), (-1,-1), 10)
                    ]))
                    story.append(risk_table)

                    doc.build(story)
                    buffer.seek(0)

                    st.download_button(
                        "⬇️ Download Detailed PDF Report",
                        buffer,
                        "churn_executive_detailed_report.pdf",
                        mime="application/pdf"
                    )

            with col2:
                if PPT_AVAILABLE:
                    if st.button("📊 Generate Investor PowerPoint"):
                        prs = Presentation()
                        slide = prs.slides.add_slide(prs.slide_layouts[0])
                        slide.shapes.title.text = "Churn Analysis Pro"
                        subtitle = slide.placeholders[1]
                        subtitle.text = "AI-Powered Customer Retention Insights"

                        slide2 = prs.slides.add_slide(prs.slide_layouts[1])
                        slide2.shapes.title.text = "Key Metrics"
                        tf = slide2.shapes.placeholders[1].text_frame
                        tf.add_paragraph().text = f"• Total Customers: {len(df):,}"
                        tf.add_paragraph().text = f"• Predicted Churn Rate: {df['predicted_churn'].mean():.1%}"
                        tf.add_paragraph().text = f"• Annual Revenue at Risk: ₦{total_revenue_at_risk:,.0f}"
                        tf.add_paragraph().text = f"• LTV at Risk: ₦{total_ltv_at_risk:,.0f}"

                        ppt_buffer = io.BytesIO()
                        prs.save(ppt_buffer)
                        ppt_buffer.seek(0)
                        st.download_button("⬇️ Download PPT", ppt_buffer, "churn_presentation.pptx",
                                         "application/vnd.openxmlformats-officedocument.presentationml.presentation")
                else:
                    st.info("PPT export requires local installation of python-pptx")

        # TAB 5: Full Data
        with tab5:
            st.download_button(
                "⬇️ Download Enriched CSV (with predictions, LTV, etc.)",
                df.to_csv(index=False).encode(),
                "churn_analysis_full_enriched.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

else:
    st.info("👆 Upload a CSV file to start the analysis.")
    st.write("Required columns: `tenure_months`, `monthly_spend_ngn`, `num_purchases`, `complaints`, `support_tickets`, `app_usage_days_per_month`, `email_open_rate`, `churn`")

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • December 2025")
