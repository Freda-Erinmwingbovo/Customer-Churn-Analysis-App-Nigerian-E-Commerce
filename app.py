        # PDF Report Generation
        st.subheader("📄 Generate & Download PDF Report")
        if st.button("Generate PDF Report"):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Customer Churn Analysis Report", styles['Title']))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Summary Metrics
            elements.append(Paragraph("Key Insights", styles['Heading2']))
            data = [
                ["Metric", "Value"],
                ["Total Customers", f"{len(df):,}"],
                ["Predicted Churn Rate", f"{df['predicted_churn'].mean():.1%}"],
                ["Annual Revenue at Risk", f"₦{revenue_risk:,.0f}"],
                ["Estimated Net Revenue Saved (What-If)", f"₦{net:,.0f}"]
            ]
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))

            # High-Risk Customers Table
            elements.append(Paragraph("High-Risk Customers", styles['Heading2']))
            high_risk_data = [["ID", "Name", "Spend (₦)", "Probability"]] + \
                             high_risk[['customer_id', 'name', 'monthly_spend_ngn', 'churn_probability']].head(20).values.tolist()
            high_table = Table(high_risk_data)
            high_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]))
            elements.append(high_table)

            # Recommendations
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Recommendations", styles['Heading2']))
            elements.append(Paragraph(f"- Target {len(high_risk)} high-risk customers with {discount}% discount", styles['Normal']))
            elements.append(Paragraph("- Focus on reducing complaints and improving app engagement", styles['Normal']))

            doc.build(elements)
            buffer.seek(0)

            st.download_button(
                label="📥 Download PDF Report",
                data=buffer,
                file_name="churn_analysis_report.pdf",
                mime="application/pdf"
            )
