import streamlit as st
import joblib
import pandas as pd

st.title("🏠 House Price Predictor")

st.write("Enter house details to predict the price.")

try:
    model = joblib.load("models/house_price_model.pkl")
    X_train = pd.read_csv("data/processed/X_train.csv")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

sample = X_train.iloc[[0]].copy()

area = st.number_input("Living Area (sq ft)", value=1500.0)
quality = st.slider("Overall Quality", 1, 10, 5)
garage = st.number_input("Garage Cars", min_value=0, max_value=5, value=2)

if st.button("Predict Price"):
    with st.spinner("Predicting house price..."):
        if "GrLivArea" in sample.columns:
            sample["GrLivArea"] = area
        if "OverallQual" in sample.columns:
            sample["OverallQual"] = quality
        if "GarageCars" in sample.columns:
            sample["GarageCars"] = garage

        prediction = model.predict(sample)

    st.success(f"Estimated House Price: ₹{prediction[0]:,.2f}")