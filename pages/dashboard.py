import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="NZ Car Model by Region", layout="wide")
st.title("📍 NZ Car Model Distribution by Region")

df = pd.read_csv('cleaned_data.csv')
with open('nz.json', 'r') as f:
    nz_geojson = json.load(f)
df.columns = df.columns.str.lower().str.strip()

geojson_regions = [f['properties']['name'] for f in nz_geojson['features']]
region_opts = sorted(set(df['region']).intersection(set(geojson_regions)))

# --- 侧边栏选择区 ---
with st.sidebar:
    st.header("Map Controls")
    selected_region = st.selectbox('Select a region to highlight:', region_opts)
    st.markdown(
        "You can hover over the map to see the distribution of car models in each region. "
        "The blue region is currently selected."
    )

region_model = df.groupby(['region', 'model']).size().reset_index(name='count')
def make_hovertext(region):
    models = region_model[region_model['region'] == region]
    if models.empty:
        return "No data"
    txt = "<br>".join([f"{row['model']}: {row['count']}" for _, row in models.iterrows()])
    return txt
hovertext = {region: make_hovertext(region) for region in geojson_regions}

counts_map = pd.DataFrame({'region': geojson_regions})
counts_map['highlight'] = counts_map['region'].apply(lambda x: 1 if x == selected_region else 0)
counts_map['hover'] = counts_map['region'].apply(lambda x: hovertext.get(x, "No data"))

fig = px.choropleth_mapbox(
    counts_map,
    geojson=nz_geojson,
    locations='region',
    color='highlight',
    featureidkey="properties.name",
    mapbox_style="carto-positron",
    center={"lat": -41.2, "lon": 174.8},
    zoom=4.5,
    color_continuous_scale=[[0, '#eeeeee'], [1, '#0081ff']],
    opacity=0.85,
    custom_data=['hover'],
)
fig.update_coloraxes(showscale=False)
fig.update_traces(
    hovertemplate="<b>%{location}</b><br>%{customdata[0]}<extra></extra>"
)
fig.update_layout(
    margin={"r":0,"t":32,"l":0,"b":0},
    title=f"<b>Car Model Distribution - Hover to See Model Counts</b><br>Selected Region: <span style='color:#0081ff'>{selected_region}</span>",
    title_x=0.5
)

left, right = st.columns([1.8, 1.1])
with left:
    st.plotly_chart(fig, use_container_width=True)

# Only keep model and count
models_in_region = (
    df[df['region'] == selected_region]
    .groupby('model')
    .size()
    .reset_index(name='Count')
    .sort_values('Count', ascending=False)
)

with right:
    st.markdown(
        f"<h4 style='margin-bottom:10px;'>Model Popularity in <span style='color:#0081ff'>{selected_region}</span></h4>",
        unsafe_allow_html=True
    )
    st.dataframe(
        models_in_region.style.highlight_max(axis=0, subset=['Count'], color='#bbf7d0'),
        use_container_width=True,
        hide_index=True,
        height=500
    )
    st.caption("Table shows model distribution in the selected region, sorted by count.")

# --- footer notes  ---
st.markdown("""
---
<small>
- <span style='color:#0081ff;font-weight:bold;'>Blue region</span> on the map is currently selected.<br>
- Hover to see detailed car model counts for each region.<br>
- Table on the right highlights the most popular model(s).
</small>
""", unsafe_allow_html=True)
