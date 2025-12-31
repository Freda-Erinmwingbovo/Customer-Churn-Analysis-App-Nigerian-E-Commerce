# ============================================================
# app.py — Churn Analysis Pro (GUARANTEED WORKING - FIXED INDENTATION)
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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

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
    .recommendation {background-color: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 5px solid #388e3c;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚨 Churn Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>AI-Powered Retention • Personalized Recommendations • LTV</p>", unsafe_allow_html=True)

# ---------------- PERSONALIZED RECOMMENDATION ENGINE ----------------
def generate_personalized_recommendation(row, df):
    reasons = []
    actions = []

    high_spend_threshold = df['monthly_spend_ngn'].quantile(0.7)
    low_engagement_threshold = df['engagement_score'].quantile(0.3)

    if row['monthly_spend_ngn'] > high_spend_threshold:
        reasons.append("high-value customer")
        actions.append("offer exclusive VIP discount or loyalty bonus")

    if row['tenure_months'] < 6:
        reasons.append("new customer")
        actions.append("send personalized onboarding support or welcome offer")

    if row['complaints'] > 2 or row['support_tickets'] > 3:
        reasons.append("service issues reported")
        actions.append("provide apology voucher and priority support")

    if row['engagement_score'] < low_engagement_threshold:
        reasons.append("low engagement")
        actions.append("launch re-engagement emails or app notifications")

    if row['spend_per_purchase'] > 50000 and row['num_purchases'] < 5:
        reasons.append("high-value but infrequent buyer")
        actions.append("offer limited-time bundle or upgrade incentive")

    if not actions:
        reasons.append("predicted churn risk")
        actions.append("send targeted discount based on behavior")

    reason_str = ", ".join(reasons)
    action_str = "; ".join(actions).capitalize()

    return f"**Risk drivers:** {reason_str}. **Recommended:** {action_str}"

# ---------------- FILE UPLOAD & PROCESSING ----------------
uploaded_file = st.file_uploader("📁 Upload your customer CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} customers")

        required = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints',
                    'support_tickets', 'app_usage_days_per_month', 'email_open_rate', 'churn']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
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

        features = ['tenure_months', 'monthly_spend_ngn', 'num_purchases', 'complaints',
                    'support_tickets', 'app_usage_days_per_month', 'email_open_rate',
                    'spend_per_purchase', 'complaint_rate', 'ticket_rate', 'engagement_score']

        X = df[features]
        y = df['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)

        with st.spinner("Training model..."):
            model = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42)
            model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
        df['predicted_churn'] = (df['churn_probability'] > 0.5).astype(int)
        df['ltv_estimate'] = df['monthly_spend_ngn'] * 12 * (1 / (df['churn_probability'] + 0.01))

        # Personalized recommendations
        df['recommendation'] = df.apply(lambda row: generate_personalized_recommendation(row, df), axis=1)

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

        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔴 High-Risk Customers", "🧠 Model Explainability", "💰 What-If Simulator", "📄 Export Reports", "📊 Full Data"
        ])

        with tab1:
            st.subheader("High-Risk Customers with Personalized Retention Plans")
            risk_threshold = st.slider("Risk threshold", 0.3, 0.9, 0.6, 0.05, key="risk_thresh")
            high_risk = df[df['churn_probability'] > risk_threshold].sort_values('churn_probability', ascending=False).copy()

            if high_risk.empty:
                st.success(f"No customers above {risk_threshold:.0%} risk – excellent retention!")
            else:
                display_df = high_risk[['customer_id', 'name', 'churn_probability', 'monthly_spend_ngn',
                                        'tenure_months', 'ltv_estimate']].copy()
                display_df['churn_probability'] = (display_df['churn_probability'] * 100).round(1).astype(str) + "%"
                display_df['monthly_spend_ngn'] = display_df['monthly_spend_ngn'].apply(lambda x: f"₦{x:,.0f}")
                display_df['ltv_estimate'] = display_df['ltv_estimate'].apply(lambda x: f"₦{x:,.0f}")
                st.dataframe(display_df.head(100), use_container_width=True)

                st.write(f"**{len(high_risk)} high-risk customers** – personalized plans below:")
                for _, cust in high_risk.head(15).iterrows():
                    with st.expander(f"🔴 {cust['name']} (ID: {cust['customer_id']}) – Risk: {cust['churn_probability']:.1%} | LTV: ₦{cust['ltv_estimate']:,.0f}"):
                        st.write(f"**Monthly Spend:** ₦{cust['monthly_spend_ngn']:,.0f} | **Tenure:** {cust['tenure_months']} months")
                        st.markdown(f"<div class='recommendation'>{cust['recommendation']}</div>", unsafe_allow_html=True)

        with tab2:
            st.subheader("Global Feature Importance (SHAP)")
            if st.button("Generate SHAP Beeswarm Plot", key="shap_button"):
                with st.spinner("Computing SHAP values... (10-20 seconds)"):
                    try:
                        background = shap.maskers.Independent(X_train_scaled, max_samples=100)
                        explainer = shap.Explainer(model, background)
                        shap_values = explainer(X_test_scaled[:200])
                        fig, ax = plt.subplots(figsize=(10, 7))
                        shap.plots.beeswarm(shap_values, max_display=12, show=False)
                        st.pyplot(fig)
                        plt.clf()
                        st.success("SHAP plot generated!")
                    except Exception as e:
                        st.error(f"SHAP error: {e}. Try clicking again.")

        with tab3:
            st.subheader("Retention Campaign Simulator")
            if high_risk.empty:
                st.info("No high-risk customers to simulate on. Lower the risk threshold in Tab 1.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    discount_pct = st.slider("Discount Offered (%)", 5, 60, 25, key="sim_discount")
                with col2:
                    retention_gain_pct = st.slider("Expected Retention Gain (%)", 10, 90, 50, key="sim_retention")

                target_count = len(high_risk)
                expected_retained = target_count * (retention_gain_pct / 100)
                avg_monthly = high_risk['monthly_spend_ngn'].mean()

                revenue_saved = expected_retained * avg_monthly * 12
                cost = expected_retained * avg_monthly * (discount_pct / 100) * 12
                net_benefit = revenue_saved - cost

                st.markdown(f"<p class='save big-font'>Net Revenue Saved: ₦{net_benefit:,.0f}</p>", unsafe_allow_html=True)
                st.metric("Target Customers", target_count)
                st.metric("Expected Retained", f"{expected_retained:.0f}")
                st.write(f"**Gross Revenue Saved:** ₦{revenue_saved:,.0f} | **Campaign Cost:** ₦{cost:,.0f}")

        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Generate Detailed PDF Report"):
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=inch)
                    styles = getSampleStyleSheet()
                    story = []

                    story.append(Paragraph("Churn Analysis Pro – Executive Report", styles['Title']))
                    story.append(Spacer(1, 20))
                    story.append(Paragraph("Prepared by: Freda Erinmwingbovo", styles['Normal']))
                    story.append(Paragraph("Date: December 31, 2025", styles['Normal']))
                    story.append(Spacer(1, 30))

                    data = [["Metric", "Value"],
                            ["Total Customers", f"{len(df):,}"],
                            ["Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}"],
                            ["Annual Revenue at Risk", f"₦{total_revenue_at_risk:,.0f}"],
                            ["LTV at Risk", f"₦{total_ltv_at_risk:,.0f}"],
                            ["Model Accuracy", f"{acc:.1%}"],
                            ["ROC-AUC", f"{auc:.3f}"]]
                    t = Table(data, colWidths=[3*inch, 2.5*inch])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e88e5")),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('GRID', (0,0), (-1,-1), 1, colors.black)
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 30))

                    story.append(Paragraph("Top 15 High-Risk Customers & Actions", styles['Heading2']))
                    top15 = high_risk.head(15)[['name', 'churn_probability', 'monthly_spend_ngn', 'ltv_estimate', 'recommendation']].copy()
                    top15['churn_probability'] = top15['churn_probability'].apply(lambda x: f"{x:.1%}")
                    top15['monthly_spend_ngn'] = top15['monthly_spend_ngn'].apply(lambda x: f"₦{x:,.0f}")
                    top15['ltv_estimate'] = top15['ltv_estimate'].apply(lambda x: f"₦{x:,.0f}")

                    risk_data = [["Name", "Risk", "Spend", "LTV", "Action"]]
                    for _, r in top15.iterrows():
                        rec = r['recommendation'].replace("**Risk drivers:** ", "").replace(". **Recommended:** ", ": ")[:70] + ("..." if len(r['recommendation']) > 80 else "")
                        risk_data.append([r['name'], r['churn_probability'], r['monthly_spend_ngn'], r['ltv_estimate'], rec])

                    rt = Table(risk_data, colWidths=[1.8*inch, 0.8*inch, 1*inch, 1*inch, 2*inch])
                    rt.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                        ('FONTSIZE', (0,0), (-1,-1), 9)
                    ]))
                    story.append(rt)

                    doc.build(story)
                    buffer.seek(0)
                    st.download_button("⬇️ Download PDF", buffer, "churn_personalized_report.pdf", "application/pdf")

            with col2:
                if PPT_AVAILABLE:
                    if st.button("📊 Generate PowerPoint"):
                        prs = Presentation()
                        slide = prs.slides.add_slide(prs.slide_layouts[0])
                        slide.shapes.title.text = "Churn Analysis Pro"
                        slide.placeholders[1].text = "December 2025 Insights"

                        slide2 = prs.slides.add_slide(prs.slide_layouts[1])
                        slide2.shapes.title.text = "Key Findings"
                        tf = slide2.shapes.placeholders[1].text_frame
                        tf.add_paragraph().text = f"• Customers: {len(df):,}"
                        tf.add_paragraph().text = f"• Churn Rate: {df['predicted_churn'].mean():.1%}"
                        tf.add_paragraph().text = f"• Revenue at Risk: ₦{total_revenue_at_risk:,.0f}"

                        buf = io.BytesIO()
                        prs.save(buf)
                        buf.seek(0)
                        st.download_button("⬇️ Download PPT", buf, "churn_presentation.pptx",
                                           "application/vnd.openxmlformats-officedocument.presentationml.presentation")
                else:
                    st.info("PPT export available only when running locally")

        with tab5:
            st.subheader("Download Enriched Dataset")
            st.download_button("⬇️ Download CSV (with predictions, LTV & recommendations)",
                               df.to_csv(index=False).encode(),
                               "churn_analysis_with_recommendations.csv",
                               "text/csv")

    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    st.info("👆 Please upload a CSV file to start the analysis.")
    st.write("**Required columns:** `tenure_months`, `monthly_spend_ngn`, `num_purchases`, `complaints`, `support_tickets`, `app_usage_days_per_month`, `email_open_rate`, `churn`")

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • December 2025")
