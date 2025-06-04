import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Car Recommendation", layout="wide")

st.title("🚙 AI Car Recommendation & Key Info")
st.markdown("""
This page recommends the best value car models within your budget, powered by a machine learning model trained on NZ used car data.
Select a model in the list below to preview its car photo.
""")

model = joblib.load('car_price_model.pkl')
df = pd.read_csv('cleaned_data.csv')
df.columns = df.columns.str.strip()
if 'Age' not in df.columns and 'Year' in df.columns:
    df['Age'] = 2025 - df['Year']

with st.sidebar:
    st.header("Recommendation Settings")
    min_price = st.number_input("Minimum Budget ($)", min_value=0, value=10000)
    max_price = st.number_input("Maximum Budget ($)", min_value=min_price, value=25000)
    fuel_opts = sorted(df['Fuel'].dropna().unique())
    fuel = st.selectbox('Fuel Type', fuel_opts)
    region_opts = sorted(df['Region'].dropna().unique())
    region = st.selectbox('Region', region_opts)
    year_opts = sorted(df['Year'].dropna().unique())
    year_opts_display = ['All'] + [str(y) for y in year_opts]
    year_sel = st.selectbox("Year (optional)", year_opts_display)
    top_n = st.slider("Top N Recommended Models", min_value=1, max_value=10, value=3, step=1)


if "recommend_df" not in st.session_state:
    st.session_state["recommend_df"] = None
if "recommend_models" not in st.session_state:
    st.session_state["recommend_models"] = []
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = None

def recommend():
    model_opts = sorted(df['Model'].dropna().unique())
    rec_combos = []
    for m in model_opts:
        mask = (
            (df['Model'] == m) &
            (df['Fuel'] == fuel) &
            (df['Region'] == region)
        )
        if year_sel != 'All':
            mask &= (df['Year'] == int(year_sel))
        sub = df[mask]
        if sub.empty:
            continue
        avg_enginesize = int(sub['EngineSize'].mean())
        avg_odometer = int(sub['Odometer'].mean())
        if 'Age' in sub.columns:
            avg_age = int(sub['Age'].mean())
        elif 'Year' in sub.columns:
            avg_age = int(2025 - sub['Year'].mean())
        else:
            avg_age = 8
        rec_combos.append({
            'Model': m,
            'AI Predicted Price': None,
            'Body Style': sub['BodyStyle'].mode().iat[0] if not sub['BodyStyle'].isnull().all() else "N/A",
            'Avg Odometer': f"{int(sub['Odometer'].mean()):,} KM",
            'Fuel': fuel,
            'Transmission': sub['Transmission'].mode().iat[0] if not sub['Transmission'].isnull().all() else "N/A",
            'Exterior Colour': sub['ExteriorColour'].mode().iat[0] if not sub['ExteriorColour'].isnull().all() else "N/A",
            'Model Detail': sub['ModelDetail'].mode().iat[0] if not sub['ModelDetail'].isnull().all() else "N/A",
            'EngineSize': avg_enginesize,
            'Region': region,
            'Age': avg_age,
            'Odometer': avg_odometer,
        })
    df_pred = pd.DataFrame(rec_combos)
    if df_pred.empty:
        st.session_state["recommend_df"] = None
        st.session_state["recommend_models"] = []
        st.session_state["selected_model"] = None
        return
    features = ['Model', 'Fuel', 'EngineSize', 'Odometer', 'Age', 'Region']
    log_pred_prices = model.predict(df_pred[features])
    pred_prices = np.exp(log_pred_prices)
    df_pred['AI Predicted Price'] = [f"${int(x):,}" for x in pred_prices]
    filtered = df_pred[(pred_prices >= min_price) & (pred_prices <= max_price)]
    if filtered.empty:
        st.session_state["recommend_df"] = None
        st.session_state["recommend_models"] = []
        st.session_state["selected_model"] = None
        return
    filtered = filtered.sort_values('AI Predicted Price').head(top_n)
    st.session_state["recommend_df"] = filtered
    st.session_state["recommend_models"] = filtered['Model'].tolist()
    st.session_state["selected_model"] = filtered['Model'].tolist()[0] if not filtered.empty else None


if st.button("🔍 Recommend Car Models"):
    recommend()

if st.session_state["recommend_df"] is not None and not st.session_state["recommend_df"].empty:
    display_cols = [
        'Model', 'AI Predicted Price', 'Body Style', 'Avg Odometer',
        'Fuel', 'Transmission', 'Exterior Colour', 'Model Detail', 'EngineSize'
    ]
    st.data_editor(
        st.session_state["recommend_df"][display_cols].reset_index(drop=True),
        column_order=display_cols,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="rec_editor",
        disabled=True,
    )
    model_choices = st.session_state["recommend_models"]
    prev_choice = st.session_state["selected_model"]
    default_index = model_choices.index(prev_choice) if prev_choice in model_choices else 0
    selected_model = st.selectbox(
        "Select a model below to preview its car photo:",
        model_choices,
        index=default_index,
        key="select_model"
    )
    st.session_state["selected_model"] = selected_model
    if selected_model:
        img_path_jpg = f"car_photo/{selected_model}.jpg"
        img_path_png = f"car_photo/{selected_model}.png"
        if os.path.exists(img_path_jpg):
            st.image(img_path_jpg, caption=f"{selected_model}", width=300)
        elif os.path.exists(img_path_png):
            st.image(img_path_png, caption=f"{selected_model}", width=300)
        else:
            st.info("No image available for this model.")

st.markdown("""
---
:information_source: **Tip:**  
Select a model in the list below the table to preview car photo below.  
Results are based on your selected filters and AI model predictions.
""")
