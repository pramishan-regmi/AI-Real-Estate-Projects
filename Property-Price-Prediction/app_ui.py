# frontend/app_ui.py
import streamlit as st
import requests
import json

st.title("🏡 Property Price Prediction")

# Input fields for prediction
features = {
    "MedInc": st.number_input("Median Income (in $10,000s)", value=4.5),
    "HouseAge": st.number_input("House Age", value=25),
    "Population": st.number_input("Population", value=3000),
    "AveOccup": st.number_input("Average Occupancy", value=3.5),
    "Rooms_per_Household": st.number_input("Rooms per Household", value=5.2),
    "Bedrooms_per_Room": st.number_input("Bedrooms per Room Ratio", value=0.2),
    "Population_per_Household": st.number_input("Population per Household", value=1.8)
}

if st.button("Predict Price"):
    response = requests.post("http://127.0.0.1:8000/predict", json=features)
    result = json.loads(response.text)
    st.subheader(f"🏠 Predicted Property Price: ${result['predicted_price']}")
