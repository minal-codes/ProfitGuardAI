import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ProfitGuard AI",
    page_icon="📊",
    layout="wide"
)

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------- LOAD MODEL ----------------

model_loaded = False

try:
    model = joblib.load("models/churn_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    model_loaded = True

except Exception as e:
    st.error(f"Actual Error: {e}")

# ---------------- TITLE ----------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

div[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 15px;
}

h1 {
    color: #4F46E5;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("ProfitGuard AI")

    st.info("""
    Features:
    - Churn Prediction
    - Revenue Risk Analysis
    - Customer Segmentation
    - Downloadable Reports
    """)

    st.success("Model Accuracy: 79%")

# ---------------- LOGO & BANNER ----------------

st.image(
    "assets/banner.png",
    use_container_width=True
)

st.sidebar.image(
    "assets/logo.png",
    width=120
)

# ---------------- HERO SECTION ----------------

st.markdown("""
<div style="
padding:30px;
border-radius:20px;
background: linear-gradient(135deg,#4F46E5,#06B6D4);
color:white;
text-align:center;
margin-bottom:20px;
">

<h1>📊 ProfitGuard AI</h1>

<h3>Predict Customer Churn • Protect Revenue • Improve Retention</h3>

<p>
AI-powered analytics platform for customer retention and revenue protection.
</p>

</div>
""", unsafe_allow_html=True)

st.subheader("Business Risk Overview")

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

    if st.button("Run Churn Analysis"):

        if not model_loaded:
            st.stop()

        data = uploaded_df.copy()

        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

        data["TotalCharges"] = data["TotalCharges"].fillna(
            data["TotalCharges"].median()
        )

        if "customerID" in data.columns:
            data = data.drop("customerID", axis=1)

        data_encoded = pd.get_dummies(
            data,
            drop_first=True
        )

        data_encoded = data_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        predictions = model.predict(data_encoded)

        uploaded_df["Predicted_Churn"] = predictions

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

        risk_customers = risk_customers.sort_values(
    by="MonthlyCharges",
    ascending=False
)

        st.dataframe(risk_customers.head(20))

        st.warning(
            f"{high_risk} customers may churn, putting approximately "
            f"${monthly_revenue_risk:,.0f} monthly revenue at risk."
)

        # ---------------- CHARTS ----------------

        chart_data = pd.DataFrame({
           "Category": [
             "Safe Customers",
             "High Risk Customers"
      ],
        "Count": [
        total_customers - high_risk,
        high_risk
    ]
    })

        st.subheader("Customer Risk Distribution")

        st.bar_chart(
          chart_data.set_index("Category")
       )
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(2,2))

        ax.pie(
         [total_customers - high_risk, high_risk],
         labels=["Safe", "High Risk"],
         autopct="%1.1f%%"
        )

        st.subheader("Customer Risk Percentage")
        st.pyplot(fig)

        st.subheader("Revenue Impact")

        revenue_df = pd.DataFrame({
            "Metric": [
               "Monthly Revenue Risk",
               "Annual Revenue Risk"
        ],
          "Value": [
              monthly_revenue_risk,
              annual_revenue_risk
        ]
   })

        st.bar_chart(
        revenue_df.set_index("Metric")
   )

# ---------------- DOWNLOAD REPORT ----------------

        csv = risk_customers.to_csv(index=False)

        st.download_button(
    label="📥 Download High Risk Customers Report",
    data=csv,
    file_name="high_risk_customers.csv",
    mime="text/csv"
)

# ---------------- SUMMARY ----------------

        st.subheader("Analysis Summary")

        st.info(
    f"""
    Total Customers: {total_customers}

    High Risk Customers: {high_risk}

    Risk Percentage: {risk_percent}%

    Monthly Revenue At Risk: ${monthly_revenue_risk:,.0f}
    """
)

# ---------------- BUSINESS INSIGHTS ----------------

st.header("Business Insights")

st.write("""
- Month-to-month customers have the highest churn risk.
- Long-term contracts significantly reduce churn.
- High-value customers contribute most revenue loss.
- Retention campaigns should target high-risk customers first.
""")