import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="ProfitGuard AI",
    page_icon="📊",
    layout="wide"
)

# Load Trained Model
try:
    model = joblib.load("models/churn_model.pkl")
    model_loaded = True
except:
    model_loaded = False

# Title
st.title("📊 ProfitGuard AI")
st.subheader("AI-Powered Customer Churn Prevention System")

# Metrics
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

# Upload CSV
st.header("Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(uploaded_df.head())

    st.subheader("Dataset Columns")
    st.write(uploaded_df.columns.tolist())

    st.subheader("Number of Rows")
    st.write(len(uploaded_df))

    if st.button("Run Churn Analysis"):
        total_customers = len(uploaded_df)

        st.success("Analysis Completed")

        st.metric(
        "Uploaded Customers",
        total_customers
    )

        st.info(
        "Prediction engine will be connected in the next step."
    )

# Model Status
st.header("Model Status")

if model_loaded:
    st.success("Churn Prediction Model Loaded Successfully")
else:
    st.error("Model File Not Found")

# Business Insights
st.header("Business Insights")

st.write("""
- Month-to-month customers have the highest churn risk.
- Long-term contracts significantly reduce churn.
- High-value customers contribute most revenue loss.
- Retention campaigns should target high-risk customers first.
""")