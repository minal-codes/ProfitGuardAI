import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ProfitGuard AI",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

try:
    model = joblib.load("models/churn_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    model_loaded = True
except Exception as e:
    model_loaded = False

# ---------------- TITLE ----------------

st.title("📊 ProfitGuard AI")
st.subheader("AI-Powered Customer Churn Prevention System")

# ---------------- MODEL STATUS ----------------

if model_loaded:
    st.success("✅ Churn Prediction Model Loaded Successfully")
else:
    st.error("❌ Model File Not Found")

# ---------------- FILE UPLOAD ----------------

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

    # ---------- RUN ANALYSIS BUTTON ----------

    if st.button("Run Churn Analysis"):

        data = uploaded_df.copy()

        # Clean TotalCharges
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

        data["TotalCharges"] = data["TotalCharges"].fillna(
            data["TotalCharges"].median()
        )

        # Remove customerID
        if "customerID" in data.columns:
            data = data.drop("customerID", axis=1)

        # One Hot Encoding
        data_encoded = pd.get_dummies(
            data,
            drop_first=True
        )

        # Match training columns
        data_encoded = data_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Predictions
        predictions = model.predict(data_encoded)

        uploaded_df["Predicted_Churn"] = predictions

        # Metrics
        total_customers = len(uploaded_df)

        high_risk = int(predictions.sum())

        risk_percent = round(
            (high_risk / total_customers) * 100,
            2
        )

        monthly_revenue_risk = uploaded_df.loc[
            uploaded_df["Predicted_Churn"] == 1,
            "MonthlyCharges"
        ].sum()

        annual_revenue_risk = monthly_revenue_risk * 12

        st.success("Analysis Completed Successfully")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Customers",
                total_customers
            )

        with col2:
            st.metric(
                "High Risk Customers",
                high_risk
            )

        with col3:
            st.metric(
                "Risk Percentage",
                f"{risk_percent}%"
            )

        st.metric(
            "Monthly Revenue At Risk",
            f"${monthly_revenue_risk:,.0f}"
        )

        st.metric(
            "Annual Revenue At Risk",
            f"${annual_revenue_risk:,.0f}"
        )

        st.subheader("High Risk Customers")

        risk_customers = uploaded_df[
            uploaded_df["Predicted_Churn"] == 1
        ]

        st.dataframe(
            risk_customers.head(20)
        )

# ---------------- BUSINESS INSIGHTS ----------------

st.header("Business Insights")

st.write("""
- Month-to-month customers have the highest churn risk.
- Long-term contracts significantly reduce churn.
- High-value customers contribute most revenue loss.
- Retention campaigns should target high-risk customers first.
""")