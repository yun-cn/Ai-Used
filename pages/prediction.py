import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚘 Car Price Prediction")

model = joblib.load('car_price_model.pkl')

with st.sidebar:
    st.header("Input Car Details")
    year = st.number_input('Year', min_value=1990, max_value=2025, value=2019)
    odometer = st.number_input('Odometer (km)', min_value=0, max_value=500_000, value=100_000)
    enginesize = st.number_input('Engine Size (cc)', min_value=800, max_value=6000, value=1800)
    model_opts = ['Corolla', 'Hiace', 'Prius', 'Hilux', 'Camry', 'Estima', 'Aqua',
                  'Land Cruiser Prado', 'RAV4', 'C-HR', 'Yaris', 'Vitz',
                  'Highlander', 'Vellfire', 'Alphard']
    model_name = st.selectbox('Model', model_opts)
    fuel_opts = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Plug-in hybrid']
    fuel = st.selectbox('Fuel Type', fuel_opts)
    region_opts = [
        'Auckland', 'Bay of Plenty', 'Canterbury', 'Gisborne', "Hawke's Bay", 'Manawatu',
        'Marlborough', 'Nelson Bays', 'Northland', 'Otago', 'Southland', 'Taranaki',
        'Timaru - Oamaru', 'Waikato', 'Wairarapa', 'Wellington', 'Whanganui', 'Not Specified'
    ]
    region = st.selectbox('Region', region_opts)

age = 2025 - year
is_hybrid = 'Yes' if fuel in ['Hybrid', 'Plug-in hybrid'] else 'No'

st.markdown(
    "#### Predict the fair market value of a used car based on key parameters. Click **Predict Price** to see the estimate:")

col1, col2, col3 = st.columns([2, 2, 2])
with col2:
    predict_btn = st.button('🔮 Predict Price', use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame({
        'Age': [age],
        'Odometer': [odometer],
        'EngineSize': [enginesize],
        'Model': [model_name],
        'Fuel': [fuel],
        'Region': [region],
        'IsHybrid': [is_hybrid]
    })

    pred_log_price = model.predict(input_data)[0]
    pred_price = np.exp(pred_log_price)
    lower = pred_price * 0.90
    upper = pred_price * 1.10

    st.markdown(
        f"""
        <div style='background: linear-gradient(90deg,#f7fafc 60%,#e3fcec 100%); padding:28px 8px 18px 8px; border-radius:18px; 
            box-shadow: 0 4px 18px #b7d6c4; text-align:center; margin-bottom:12px;'>
            <span style='font-size:2.7em; font-weight:bold; color:#1b5e20;'>💰 ${pred_price:,.0f}</span>
            <br>
            <span style='font-size:1.14em; color:#333;'>Predicted Market Price</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style='background-color:#e0f7fa; padding:16px 6px 14px 6px; border-radius:12px; text-align:center; margin-bottom:12px;'>
            <span style='font-size:1.08em; color:#0277bd; font-weight:600;'>Estimated Price Range:</span>
            <br>
            <span style='font-size:1.6em; color:#00bfae; font-weight:bold;'>${lower:,.0f}  ~  ${upper:,.0f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.99em; margin-top:5px;'>"
        "Note: The price range is for reference only. Actual value may vary with market trends and vehicle condition."
        "</div>",
        unsafe_allow_html=True
    )

    with st.expander("Show input details"):
        st.write(input_data.T.rename(columns={0: "Your Input"}))
