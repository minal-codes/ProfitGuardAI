import streamlit as st

st.set_page_config(
    page_title="ProfitGuard AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ProfitGuard AI")

st.subheader("Customer Churn Risk Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", "7043")

with col2:
    st.metric("High Risk Customers", "1760")

with col3:
    st.metric("Monthly Revenue At Risk", "$131,266")

annual_loss = 131266.15 * 12

st.metric(
    "Annual Revenue At Risk",
    f"${annual_loss:,.0f}"
)

st.header("Business Insights")

st.write("""
- Month-to-month customers have the highest churn risk.
- High-paying customers contribute most revenue loss.
- Long-term contracts significantly reduce churn.
- Retention campaigns should focus on high-value customers.
""")