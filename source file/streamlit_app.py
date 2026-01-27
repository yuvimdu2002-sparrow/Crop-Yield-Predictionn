        
import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

model=load_model("Crop_model.h5",compile=False)
Soil_enc=joblib.load("soil_enc.pkl")
Crop_enc=joblib.load("crop_enc.pkl")
Weather_enc=joblib.load("weather_en.pkl")
Fertilizer_enc=joblib.load("fertilizer_enc.pkl")
Irrigation_enc=joblib.load("irrigation_enc.pkl")
scaler=joblib.load("scale.pkl")

st.markdown("<h1 style='text-align:center;'>ASMLY App</h1>", unsafe_allow_html=True)

st.subheader("🌾 Crop Yield Prediction")


import base64
import streamlit as st

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpg")  # correct path here
import pandas as pd

import streamlit as st
import pandas as pd

st.caption("Dataset Selection")

# Tabs
tab1, tab2 = st.tabs(["Data Input", "Data Preview"])

with tab1:
    option = st.radio(
        "Choose data source",
        ["Upload CSV", "Use sample data"]
    )

    df = None

    if option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Uploaded successfully ✅")

    else:
        try:
            df = pd.read_csv("crop_yield.csv")
            st.info("Using sample dataset 📄")
        except FileNotFoundError:
            st.error("sample_data.csv not found")

with tab2:
    st.subheader("Dataset Preview")
    if df is not None:
        st.dataframe(df)
    else:
        st.warning("No data loaded yet")


st.sidebar.write("Input Parameter")
Soil_Type=st.sidebar.selectbox("1.Type of the soil: ",['Chalky','Clay','Loam','Peaty','Sandy','Silt']).strip().capitalize()
Crop_Type=st.sidebar.selectbox("2.Which crop I can feed: ",['Barley','Cotton','Maize','Rice','Soybean','Wheat']).strip().capitalize()
Rainfall_mm=st.sidebar.number_input("3.Rainfall in mm: ",min_value=0, max_value=5000)
Temperature_Celsius=st.sidebar.number_input("4.Temperature: ",min_value=0, max_value=5000)
Weather_Condition=st.sidebar.selectbox("5.Weather_Condition: ",['Cloudy','Rainy','Sunny'])
Fertilizer_Use=st.sidebar.selectbox("6.Fertilizer_Used: ",["False",  "True"]).strip().capitalize()
Irrigation_Use=st.sidebar.selectbox("7.Irrigation_Used: ",["False", "True"]).strip().capitalize()

Fertilizer_Used = 'True' if Fertilizer_Use == 'Yes' else 'False'
Irrigation_Used = 'True' if Irrigation_Use == 'Yes' else 'False'

Soil_Type_enc=Soil_enc.transform([Soil_Type])
Weather_Condition_enc=Weather_enc.transform([Weather_Condition])
Fertilizer_Used_enc=Fertilizer_enc.transform([Fertilizer_Use])
Irrigation_Used_enc=Irrigation_enc.transform([Irrigation_Use])
Crop_encode=Crop_enc.transform([Crop_Type])

input_data=np.array([[Soil_Type_enc[0],Crop_encode[0],Rainfall_mm,Temperature_Celsius,Weather_Condition_enc[0],Fertilizer_Used_enc[0],Irrigation_Used_enc[0]]])
scaledin_data=scaler.transform(input_data)

if st.button("Predict"):
  pred=model.predict(scaledin_data)[0]
  st.write('Predicted Yield 🌽🧺')
  st.metric(label="Yield (tons Per Hectare)",value=f"{pred[0]:.3f}")













