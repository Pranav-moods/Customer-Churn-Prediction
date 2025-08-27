import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("anaconda_projects_4c57465e-5ed5-4f90-a8ea-49eef1ac14c4_churn_pipeline.pkl")   # your trained pipeline model

st.title("Customer Churn Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.write(data.head())

    try:
        # Make predictions
        predictions = model.predict(data)
        data["Churn_Prediction"] = predictions

        st.write("### Predictions")
        st.write(data[["customerID", "Churn_Prediction"]])

        # Download link for results
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error during prediction: {e}")
