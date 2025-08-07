# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 21:08:55 2025

@author: 31908
"""
pip install xlrd==2.1.0

import pandas as pd
import numpy as np
import glob
import os

# Set your folder path
folder_path = "D:\Mortality"

# Get all .xls files in the folder
xls_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

print(f"Found {len(xls_files)} files.")

# List to store DataFrames
df_list = []

# Loop through and read each .xls file
for file in xls_files:
    try:
        df = pd.read_excel(file)  # Specify 'xlrd' for .xls
        df_list.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine all into one DataFrame
cdc_wonder = pd.concat(df_list, ignore_index=True)

cdc_wonder.dtypes

cdc_wonder.Year.nunique()

cdc_wonder = cdc_wonder.drop(columns = ['Notes','Age Group Code','Year Code'])

# ✅ Convert Deaths and Population to numeric, coerce errors to NaN
cdc_wonder['Deaths'] = pd.to_numeric(cdc_wonder['Deaths'], errors='coerce')
cdc_wonder['Population'] = pd.to_numeric(cdc_wonder['Population'], errors='coerce')

# Group by Year to get total Deaths and Population
totals = cdc_wonder.groupby(['Year','County']).agg({
    'Deaths': 'sum',
    'Population': 'sum'
}).rename(columns={
    'Deaths': 'Total_Deaths',
    'Population': 'Total_Population'
}).reset_index()

# Use pivot_table with aggregation
# Split the data based on Year
pre_2017 = cdc_wonder[cdc_wonder['Year'] < 2017]
post_2017 = cdc_wonder[cdc_wonder['Year'] >= 2017]

# Pivot pre-2017 data using 'Age Group'
pivot_pre_2017 = pre_2017.pivot_table(
    index=['Year', 'County', 'County Code'],
    columns='Age Group',
    values=['Deaths', 'Population', 'Crude Rate'],
    aggfunc='sum'
)

# Pivot post-2017 data using 'Ten Year Age Groups'
pivot_post_2017 = post_2017.pivot_table(
    index=['Year', 'County', 'County Code'],
    columns='Ten-Year Age Groups',
    values=['Deaths', 'Population', 'Crude Rate'],
    aggfunc='sum'
)

# Concatenate both pivoted results
cdc_wonder_pivoted = pd.concat([pivot_pre_2017, pivot_post_2017])

cdc_wonder_pivoted.columns = [f"{var}_{age}" for var, age in cdc_wonder_pivoted.columns]
cdc_wonder_pivoted = cdc_wonder_pivoted.reset_index()

cdc_wonder_pivoted.columns

os.getcwd()
result = cdc_wonder_pivoted.merge(totals, on=['Year','County'])

result = result.drop(columns = 'Year_y')
result = result.rename(columns = {'Year_x':'Year'})

result.to_csv('cdc_wonder.csv',index = False)













