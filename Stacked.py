# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 23:05:07 2025

@author: 31908
"""

#Treatment/Control
import plotly.express as px
import pandas as pd
import json
import os

stacked_match = pd.read_csv(r"C:\Users\31908\Downloads\stacked_match.csv")

# Initialize dictionaries to store results
treated_counties = {}
control_counties = {}

# Loop over each sub_exp year
for sub_exp_year in stacked_match['sub_exp'].unique():
    # Filter for this sub_exp group
    df_subset = stacked_match[stacked_match['sub_exp'] == sub_exp_year]
    
    # Counties treated in or after sub_exp (in post period)
    treated = df_subset[(df_subset['treat'] == 1) & (df_subset['year'] >= sub_exp_year)]
    treated_counties[sub_exp_year] = treated['county'].unique().tolist()
    
    # Counties in control: either never treated or treated after this sub_exp
    control = df_subset[(df_subset['treat'] == 0) | (df_subset['year'] < sub_exp_year)]
    control_counties[sub_exp_year] = control['county'].unique().tolist()

# Example: show treated/control for 1995
print("Treated counties for sub_exp=1995:", treated_counties[2023])
print("Control counties for sub_exp=1995:", control_counties[2023])

# Ensure a FIPS column exists (5-digit str combining state & county codes)
stacked_match['FIPS'] = stacked_match['county'].astype(str).str.zfill(5)

# For each year, generate a color variable (0 = control, 1 = treated)
years = sorted(stacked_match['year'].unique())

# Download US county GeoJSON (example link, check for the most recent version)
with open(r"C:\Users\31908\Desktop\us_counties.geojson.json") as f:
    counties = json.load(f)

fig = px.choropleth(
    stacked_match,
    geojson=counties,
    locations='FIPS',
    color='treat',
    animation_frame='year',   # makes it interactive over years!
    hover_name='county',
    hover_data=['sub_exp','post'],
    scope='usa',
    color_continuous_scale=['#e0e0e0','#21439c'],  # control: grey, treated: blue
    category_orders={'year': years}
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Treatment vs Control Counties by Year", margin={"r":0,"t":50,"l":0,"b":0})
fig.show()

years = sorted(stacked_match['year'].unique())
os.makedirs('frames', exist_ok=True)

for year in years:
    fig2 = px.choropleth(
        stacked_match[stacked_match['year'] == year],
        geojson=counties,
        locations='FIPS',
        color='treat',
        hover_name='county',
        hover_data=['sub_exp', 'post'],
        scope='usa',
        color_continuous_scale=['#e0e0e0', '#21439c']
    )
    fig2.update_geos(fitbounds="locations", visible=False)
    fig2.update_layout(title_text=f"Year: {year}", margin={"r":0,"t":50,"l":0,"b":0})
    fig2.write_image(f"frames/frame_{year}.png")  # Requires 'kaleido'

pip install -U kaleido

# Combine PNGs into GIF or MP4
import imageio
images = [imageio.imread(f"frames/frame_{year}.png") for year in years]
imageio.mimsave("treatment_control_map.gif", images, duration=1)  # 1 second per frame

# Save interactive animation to an HTML file
fig.write_html("treatment_control_interactive_map.html")
print("Interactive map saved at treatment_control_interactive_map.html")
#############################################################################################
import pandas as pd
import json
import geopandas as gpd
import plotly.express as px

# 1. Load your treatment/control data
df = stacked_match
df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)

# 2. Load and simplify US counties GeoJSON
geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
# Read with geopandas for geometry simplification
counties_gdf = gpd.read_file(geojson_url)
# 3. Filter GeoDataFrame to only relevant counties for efficiency (and avoid breaking the map)
usable_fips = df['FIPS']
filtered_gdf = counties_gdf[counties_gdf['id'].isin(usable_fips)].copy()
filtered_gdf['geometry'] = filtered_gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)
filtered_geojson = json.loads(filtered_gdf.to_json())

# 4. Plot
fig = px.choropleth(
    df,
    geojson=filtered_geojson,        # ← filtered, pre-simplified GeoJSON
    locations='county',
    color='treat',
    animation_frame='sub_exp',
    hover_name='county',
    hover_data=['sub_exp', 'post'],
    scope='usa',
    color_continuous_scale=['#e0e0e0', '#21439c'],
    height=600,
    width=900
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(
    title_text="Treatment vs Control Counties by Year",
    margin=dict(r=0, t=40, l=0, b=0)
)
fig.write_html("treatment_control_interactive_map.html")
###########################################################################
import pandas as pd
import geopandas as gpd
import json
from urllib.request import urlopen
import plotly.express as px

rows = []
for year, treated_list in treated_counties.items():
    for c in treated_list:
        rows.append({'year': year, 'FIPS': str(c).zfill(5), 'treat': 1})
for year, control_list in control_counties.items():
    for c in control_list:
        rows.append({'year': year, 'FIPS': str(c).zfill(5), 'treat': 0})

df_counties = pd.DataFrame(rows)

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties_gdf = json.load(response)

usable_fips = df_counties['FIPS'].unique()
# Convert treat to string so Plotly treats it as categorical
df_counties['treat'] = df_counties['treat'].astype(str)

fig = px.choropleth(
    df_counties,
    geojson=counties_gdf,
    locations='FIPS',
    color='treat',
    animation_frame='year',
    hover_name='FIPS',
    scope='usa',
    color_discrete_map={
       '0': '#e0e0e0',  # control: gray
       '1': '#21439c'   # treated: blue
   },  # gray for control, blue for treated
    height=500,
    width=800,
    category_orders={'treat':['0','1']}
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(
    title_text="Treated and Control Counties Over Years",
    margin=dict(r=0, t=40, l=0, b=0)
)
fig.write_html("treated_control_counties_map.html")