# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 12:05:56 2025

@author: 31908
"""
import pandas as pd
import numpy as np
import glob
import os

# Set your folder path
folder_path = r"D:\F5500N"

# Find all CSV files in that folder
file_list = glob.glob(os.path.join(folder_path, "*.csv"))

# Read and combine all CSVs into one DataFrame
df_list = []

for file in file_list:
    print(f"Reading {file}...")
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all into a single DataFrame
df_combined = pd.concat(df_list, ignore_index=True)

print("Combined shape:", df_combined.shape)

df_ins = df_combined

df_ins.columns

df_ins.dtypes


df_ins['LAST_RPT_SPONS_EIN'] = df_ins['LAST_RPT_SPONS_EIN'].apply(
    lambda x: str(int(x)).zfill(9) if pd.notna(x) else np.nan
)

df_ins[['LAST_RPT_SPONS_EIN','SPONS_DFE_EIN']].head()

df_ins[['LAST_RPT_SPONS_EIN']].isna().sum()

df_ins['EIN_formatted'] = np.where(
    df_ins['LAST_RPT_SPONS_EIN'].notna(),
    df_ins['LAST_RPT_SPONS_EIN'].astype(str).str.slice(0, 2) + '-' + df_ins['LAST_RPT_SPONS_EIN'].astype(str).str.slice(2),
    np.nan
)

df_ins[['SPONSOR_DFE_NAME']].nunique()

# 1️⃣ Group by EIN and count unique sponsor names
name_counts = (
    df_ins.groupby('SPONS_DFE_EIN')['SPONSOR_DFE_NAME']
    .nunique()
    .reset_index(name='name_count')
)

# 2️⃣ Filter EINs with more than 1 unique sponsor name
changed_eins = name_counts[name_counts['name_count'] > 1]['SPONS_DFE_EIN']

df_ins['LAST_RPT_SPONS_EIN'].nunique()
df_ins['PREV_EIN'] = df_ins.groupby('EIN_formatted')['SPONS_DFE_EIN'].shift(1)
# 3. Check where ein changed
df_ins['EIN_CHANGED'] = (df_ins['SPONS_DFE_EIN'] != df_ins['PREV_EIN']) & df_ins['PREV_EIN'].notna()

df_ins_change_EIN = df_ins[df_ins['EIN_CHANGED'] == True]


df_ins['LAST_RPT_SPONS_EIN'].isin(changed_eins)

# 2. Get previous sponsor name
df_ins['PREV_NAME'] = df_ins.groupby('SPONS_DFE_EIN')['SPONSOR_DFE_NAME'].shift(1)

# 3. Check where name changed
df_ins['NAME_CHANGED'] = (df_ins['SPONSOR_DFE_NAME'] != df_ins['PREV_NAME']) & df_ins['PREV_NAME'].notna()

df_ins_change_names = df_ins[df_ins['NAME_CHANGED'] == True]

lst_change = pd.DataFrame(df_ins_change_names.SPONS_DFE_EIN.unique())

a = lst_change[0].isin(changed_eins)

a.value_counts()
new_df = df_ins.copy()
new_df.dtypes

ein_change = new_df[new_df.SPONS_DFE_EIN != new_df.EIN_formatted]

ein_change = ein_change.dropna(subset='LAST_RPT_SPONS_EIN')

orbis_df1 = pd.read_excel(r"D:\Name_EIN 1.xlsx", sheet_name='Results')
orbis_df2 = pd.read_excel(r"D:\Name_EIN 2.xlsx", sheet_name='Results')

orbis_df = pd.concat([orbis_df1,orbis_df2])
orbis_df.nunique()

imp_df2 = pd.read_csv(r"C:\Users\31908\OneDrive - Indian School of Business\f5500_premium_data.csv")
imp_df2.spons_dfe_ein.nunique()
premium_df = imp_df2.copy()
premium_df = premium_df.drop_duplicates(subset=['spons_dfe_ein'])

imp_df = imp_df2.merge(orbis_df, left_on = 'spons_dfe_ein',right_on='Tax Identification Number (TIN)')
imp_df.spons_dfe_ein.nunique()

found_ein = imp_df.spons_dfe_ein.unique()

imp_filtered = imp_df2[~imp_df2.spons_dfe_ein.isin(found_ein)]

imp_filtered.spons_dfe_ein.nunique()

df_og = df_ins[df_ins.SPONS_DFE_EIN.isin(imp_filtered.spons_dfe_ein)]

df_og.SPONS_DFE_EIN.nunique()

b = imp_filtered[~imp_filtered.spons_dfe_ein.isin(df_og.SPONS_DFE_EIN)]

df_ins_change_names_not_found = df_ins_change_names[df_ins_change_names.SPONS_DFE_EIN.isin(imp_filtered.spons_dfe_ein)]

df_ins_change_names_not_found.to_csv('Names_Search_Orbis.csv', index = False)
imp_filtered.to_csv("D:/not_matched.csv",index=False)

match_try = pd.read_csv(r"D:\Bankruptcy.csv")
match_try.columns

match_try = match_try.merge(imp_filtered, left_on = 'SP_TAX_ID', right_on = 'spons_dfe_ein')

folder_path = "D:\CIQ"

# Get all .xlsx files in the folder
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

print(f"Found {len(excel_files)} files.")

# Read each file into a list of DataFrames
dfs = []
for file in excel_files:
    df = pd.read_excel(file, sheet_name="Sheet1")  # adjust sheet_name if needed
    df['source_file'] = os.path.basename(file)     # optional: track which file it came from
    dfs.append(df)

# Combine all DataFrames (optional)
orbis_bk  = pd.concat(dfs, ignore_index=True)
CIQ_bk.drop_duplicates()
CIQ_bk  = pd.concat(dfs, ignore_index=True)

def format_tin(x):
    if pd.isna(x):
        return np.nan
    x_str = str(x)
    # Only pad if all characters are digits
    if x_str.isdigit():
        return x_str.zfill(9)
    else:
        return x_str  # leave as-is if non-numeric

orbis_bk['Tax Identification Number (TIN)'] = orbis_bk['Tax Identification Number (TIN)'].apply(format_tin)
orbis_bk['Tax Identification Number (TIN)'] = np.where(
    orbis_bk['Tax Identification Number (TIN)'].notna(),
    orbis_bk['Tax Identification Number (TIN)'].astype(str).str.slice(0, 2) + '-' + orbis_bk['Tax Identification Number (TIN)'].astype(str).str.slice(2),
    np.nan
)
new_columns = CIQ_bk.iloc[3]

# 2️⃣ Set the columns
CIQ_bk.columns = new_columns

# 3️⃣ Drop the first 4 rows (original header row + first 3 rows)
CIQ_bk = CIQ_bk.iloc[2:].reset_index(drop=True)

trial_complete = trial_complete.merge(imp_filtered, left_on = 'SP_TAX_ID', right_on = 'spons_dfe_ein')
found_ein = np.append(found_ein,trial_complete.SP_TAX_ID.unique())

imp_filtered = imp_df2[~imp_df2.spons_dfe_ein.isin(found_ein)]

matched_prem_df_ciq = premium_df.merge(trial_complete,left_on = 'spons_dfe_ein',right_on='SP_TAX_ID')

CIQ_bk = pd.read_excel(r"C:\Users\31908\Dropbox\RA Share File 2024-25\CapitalIQ\bk.xlsx")
new_columns = CIQ_bk.iloc[3]

# 2️⃣ Set the columns
CIQ_bk.columns = new_columns

# 3️⃣ Drop the first 4 rows (original header row + first 3 rows)
CIQ_bk = CIQ_bk.iloc[4:].reset_index(drop=True)
CIQ_bk = CIQ_bk.drop_duplicates(subset = 'SP_ENTITY_NAME')
CIQ_bk = CIQ_bk.reset_index()
df_combined.nunique()
df_bk_match = premium_df.merge(CIQ_bk,left_on = 'spons_dfe_ein', right_on = 'SP_TAX_ID')
df_bk_match.spons_dfe_ein.nunique()
matched_prem_df_orbis = premium_df.merge(orbis_df,)

orbis_comp = pd.read_excel(r"D:\Orbis\Export 07_07_2025 11_19.xlsx",sheet_name='Results')
orbis_comp.columns

prem_names = premium_df.merge(orbis_df, left_on = 'spons_dfe_ein',right_on = 'Tax Identification Number (TIN)',how='left')
prem_names = prem_names.drop_duplicates(subset=['spons_dfe_ein'])

premium_CIQ_bk = prem_names.merge(CIQ_bk, left_on = 'spons_dfe_ein', right_on ='SP_TAX_ID')
premium_CIQ_bk.nunique()
both_match = premium_CIQ_bk.merge(orbis_bk, left_on='spons_dfe_ein',right_on = 'Tax Identification Number (TIN)')
orbis_matched = prem_names.merge(orbis_bk, on ='Company name Latin alphabet')
prem_names.nunique()

# 1️⃣ Merge with orbis_bk
merged_orbis = prem_names.merge(
    orbis_bk,
    on='Company name Latin alphabet',
    how='left',
    indicator='orbis_match'
)

# Convert _merge values to dummy
merged_orbis['orbis_match'] = merged_orbis['orbis_match'].apply(lambda x: 1 if x == 'both' else 0)

# 2️⃣ Merge with CIQ_bk
merged_final = merged_orbis.merge(
    CIQ_bk,
    left_on='spons_dfe_ein',
    right_on = 'SP_TAX_ID',
    how='left',
    indicator='ciq_match'
)

# Convert _merge values to dummy
merged_final['ciq_match'] = merged_final['ciq_match'].apply(lambda x: 1 if x == 'both' else 0)

# Example: remove columns ending with _x or _y after a merge
cols_to_drop = [col for col in merged_final.columns if str(col).endswith('_x') or str(col).endswith('_y')]
merged_final = merged_final.drop(columns=cols_to_drop)

unq = merged_final.nunique()

merged_final.columns

merged_final.orbis_match.value_counts()

merged_final.ciq_match.value_counts()

merged_final = merged_final[ ['tot_ins_prsn_covered_eoy_cnt','spons_dfe_ein','spons_dfe_pn','spons_dfe_state','spons_dfe_zip_code','ins3','net_partcp','health','hmo','ppo','life_ins','dental','vision',
'drug','stoploss','indem','other','tot_prem','year','uins3','temp','state','county','hrrnum','business_code','zip_code',
'hcacount','chscount','sys_pe','hosp_pe','sys_pe_add','hosp_pe_add','pe','pe_add','chsevent','eventcount','hcaevent','event',
'event2','event3','chsevent25','hcaevent25','event25','event225','forprofit','region','log_pcpi','log_population','Company name Latin alphabet','Inactive',
'Quoted','Branch','OwnData','Country ISO code','NACE Rev. 2, core code (4 digits)','Consolidation code','orbis_match','ciq_match']]

#merged_final.to_csv('all_merged.csv',index = False)

