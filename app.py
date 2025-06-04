import streamlit as st

st.set_page_config(page_title="Smart NZ Used Car PricePrediction & RecommendationPlatform", layout="wide")

st.markdown("""
<div style='padding: 18px 12px 10px 12px; background: linear-gradient(90deg, #e0e7ef 70%, #fff 100%); border-radius: 18px; margin-bottom: 22px;'>
    <h1 style='font-size:2.4em; margin-bottom:5px; color:#273469; letter-spacing:1px;'>
        🚗 Used Car Price Analysis & Prediction Dashboard
    </h1>
    <div style='font-size:1.15em; color:#555; margin-bottom:2px;'>
        <b>Welcome!</b> This interactive dashboard lets you explore, analyze, and predict New Zealand used car prices with state-of-the-art machine learning models and visual analytics.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 🧭 **Navigation**
Use the **left sidebar** to switch between the main features:
""")

# 分区卡片式导航
st.markdown("""
<div style='display: flex; gap: 32px; flex-wrap: wrap; margin-bottom:16px;'>

<div style='background:#f2f6fc; border-radius:13px; padding:18px 24px; min-width:250px; margin-bottom:10px;'>
    <span style='font-size:1.32em;'>📊 <b>Dashboard</b></span><br>
    <span style='color:#555;'>Visualise and download the data.</span>
</div>

<div style='background:#f2f6fc; border-radius:13px; padding:18px 24px; min-width:250px; margin-bottom:10px;'>
    <span style='font-size:1.32em;'>📉 <b>Depreciation Curve</b></span><br>
    <span style='color:#555;'>View how price drops over future years.</span>
</div>

<div style='background:#f2f6fc; border-radius:13px; padding:18px 24px; min-width:250px; margin-bottom:10px;'>
    <span style='font-size:1.32em;'>🤖 <b>Prediction</b></span><br>
    <span style='color:#555;'>Predict car prices using a machine learning model.</span>
</div>

<div style='background:#f2f6fc; border-radius:13px; padding:18px 24px; min-width:250px; margin-bottom:10px;'>
    <span style='font-size:1.32em;'>💡 <b>Find Car By Budget</b></span><br>
    <span style='color:#555;'>Find all cars you can buy within your budget and filters.</span>
</div>
</div>
""", unsafe_allow_html=True)

# 页脚说明
st.markdown("""
---
<small>
<b>Tip:</b> Use the sidebar on the left at any time to switch pages and try different tools.  
All insights and predictions are based on NZ used car data.
</small>
""", unsafe_allow_html=True)
