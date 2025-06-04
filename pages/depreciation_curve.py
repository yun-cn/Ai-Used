import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Future Car Depreciation", layout="wide")

st.title("🚗 Future Car Price Depreciation Prediction")
st.markdown("""
This tool predicts how your selected car's price will decrease in the coming years using a machine learning model trained on real NZ used car data.
- **Choose the car type, fuel, region and how many years ahead to see the depreciation trend and annual price drop.**
""")

model = joblib.load('car_price_model.pkl')
df = pd.read_csv('cleaned_data.csv')
df.columns = df.columns.str.strip()

with st.sidebar:
    st.header("Prediction Settings")
    model_opts = sorted(df['Model'].dropna().unique())
    selected_model = st.selectbox('Select Car Model', model_opts)
    fuel_opts = sorted(df[df['Model'] == selected_model]['Fuel'].dropna().unique())
    selected_fuel = st.selectbox('Fuel Type', fuel_opts)
    region_opts = sorted(df[(df['Model'] == selected_model) & (df['Fuel'] == selected_fuel)]['Region'].dropna().unique())
    selected_region = st.selectbox('Region', region_opts)
    years_ahead = st.slider('Years Ahead to Predict', min_value=1, max_value=15, value=6, help="Predict depreciation up to how many years in the future?")

sample = df[
    (df['Model'] == selected_model) &
    (df['Fuel'] == selected_fuel) &
    (df['Region'] == selected_region)
].copy()

if sample.empty:
    st.warning("No data for this configuration.")
else:
    enginesize = int(sample['EngineSize'].mean())
    if 'Age' not in sample.columns and 'Year' in sample.columns:
        sample['Age'] = 2025 - sample['Year']
    mean_age = sample['Age'].mean()
    odo_per_year = int(sample['Odometer'].mean() / mean_age) if mean_age > 0 else 15000

    st.markdown(f"""
    <div style='background-color:#f5f7fa; padding:12px; border-radius:8px; margin-bottom:20px;'>
        <b>Average engine size:</b> {enginesize} cc &nbsp;|&nbsp;
        <b>Average odometer per year:</b> {odo_per_year:,} km/year
    </div>
    """, unsafe_allow_html=True)

    pred_results = []
    for i in range(years_ahead+1):
        future_year = 2025 + i
        age = mean_age + i
        odometer = sample['Odometer'].mean() + odo_per_year * i
        row = {
            'Model': selected_model,
            'Fuel': selected_fuel,
            'EngineSize': enginesize,
            'Odometer': odometer,
            'Age': age,
            'Region': selected_region,
            'FutureYear': future_year
        }
        pred_results.append(row)
    df_pred = pd.DataFrame(pred_results)
    features = ['Model', 'Fuel', 'EngineSize', 'Odometer', 'Age', 'Region']
    log_pred_prices = model.predict(df_pred[features])
    pred_prices = np.exp(log_pred_prices)
    df_pred['Predicted Price'] = pred_prices

    df_pred['Annual Price Drop'] = df_pred['Predicted Price'].shift(0) - df_pred['Predicted Price'].shift(-1)
    df_pred['Annual Price Drop'] = df_pred['Annual Price Drop'].fillna(0).astype(int)

    left, right = st.columns([2,1])
    with left:
        st.subheader("Price Depreciation Trend")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df_pred['FutureYear'], df_pred['Predicted Price'], marker='o', color="#00aaff", linewidth=2)
        ax.fill_between(df_pred['FutureYear'], df_pred['Predicted Price'], color="#e6f7ff", alpha=0.7)
        for i, v in enumerate(df_pred['Predicted Price']):
            ax.text(df_pred['FutureYear'].iloc[i], v, f"${int(v):,}", fontsize=9, ha='center', va='bottom')
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted Price (NZD)")
        ax.set_title(f"{selected_model} | {selected_fuel} | {selected_region}")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with right:
        st.subheader("Depreciation Table")
        show_df = df_pred.copy()
        show_df['Predicted Price'] = show_df['Predicted Price'].apply(lambda x: f"${int(x):,}")
        show_df['Annual Price Drop'] = show_df['Annual Price Drop'].apply(lambda x: f"${int(x):,}" if x > 0 else "-")
        st.dataframe(show_df[['FutureYear', 'Predicted Price', 'Annual Price Drop']], use_container_width=True)

    st.caption(
        "Predicted prices and annual depreciation for each future year. \n"
        "Engine size and odometer are averaged from real NZ used car data."
    )
