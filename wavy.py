# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 23:31:42 2025

@author: 31908
"""
os.chdir('D:/')

import zipfile
import polars as pl
import os
import glob


os.getcwd()

# Define the folder containing your CSV files
data_folder = "D:/PPD_data/Private_Investments"

# List of all 4 CSV files (modify with your actual filenames)
csv_files = [
    "Private_Pension_Plans.csv",
    "Private_Pension_Plans2.csv",
    "Private_Pension_Plans3.csv",
    "Private_Pension_Plans4.csv"
]

# Define keywords for identifying, location, and investment variables
keywords = [
    "ACK_ID", "PLAN_YR", "PLAN_YEAR","FY_BGN_DT","FY_END_DT","DT_RECEIVED","DT_PLAN_EFFECTIVE","DT_SIGNED","PROC_DT", "PROC_DATE","FILING_DATE","PLAN_EFFECTIVE_DATE"
,"EIN", "PLAN_NAME", "PLAN_NUMBER", "SPONSOR_DFE_NAME",
    "SPONS_DFE_MAIL_US_ADDRESS1", "SPONS_DFE_MAIL_US_ADDRESS2", "SPONS_DFE_MAIL_US_CITY",
    "SPONS_DFE_MAIL_US_STATE", "SPONS_DFE_MAIL_US_ZIP", "SPONS_DFE_LOC_US_ADDRESS1",
    "SPONS_DFE_LOC_US_ADDRESS2", "SPONS_DFE_LOC_US_CITY", "SPONS_DFE_LOC_US_STATE",
    "SPONS_DFE_LOC_US_ZIP", "SPONS_DFE_FOREIGN_ADDR1", "SPONS_DFE_FOREIGN_ADDR2",
    "SPONS_DFE_FOREIGN_CITY", "SPONS_DFE_FOREIGN_PROVSTATE", "SPONS_DFE_FOREIGN_CNTRY",
    "SPONS_DFE_FOREIGN_POSTAL_CD", "SPONS_DFE_PHONE_NUM", "SPONS_DFE_NAME_SIGNER",
    "SPONS_DFE_TITLE_SIGNER", "TOT_ASSETS_BOY", "TOT_ASSETS_EOY", "TOT_LIABILITIES_BOY",
    "TOT_LIABILITIES_EOY", "NET_ASSETS_BOY", "NET_ASSETS_EOY", "SCH_H_PLAN_ASSETS_INT",
    "SCH_H_PLAN_ASSETS_CASH", "SCH_H_PLAN_ASSETS_STOCK", "SCH_H_PLAN_ASSETS_BOND",
    "SCH_H_PLAN_ASSETS_OTHER", "SCH_H_INVST_RL_ESTATE", "SCH_H_INVST_OTHER",
    "SCH_H_INVST_PARTNERSHIP", "SCH_H_INVST_HOLDINGS", "SCH_H_INVST_MTGE",
    "SCH_H_INVST_SVGS_ACCT", "SCH_H_INVST_GIC"
]

# Function to extract only matching columns from a large CSV file
def extract_columns(file_path):
    # First, scan just the header to get column names
    schema = pl.read_csv(file_path, n_rows=0).columns
    selected_columns = [col for col in schema if any(key in col.upper() for key in keywords)]

    # Read only the selected columns
    df = pl.read_csv(file_path, columns=selected_columns)
    return df

all_files = glob.glob(f"{data_folder}/*.csv")

dataframes = []

for file in all_files:
    try:
        df = pl.read_csv(file, columns=keywords, ignore_errors=True)
        df = df.select([col for col in keywords if col in df.columns])  # select only available
        df = df.with_columns([pl.lit(None).alias(col) for col in keywords if col not in df.columns])
        df = df.select(keywords)  # reorder to match
        dataframes.append(df)
    except Exception as e:
        print(f"Skipping {file}: {e}")

# Concatenate all with consistent schema
combined_df = pl.concat(dataframes, how="vertical")

combined_df.write_csv(csv_output_path)

print(f"Final combined shape: {combined_df.shape}")
# Save to a CSV first
csv_output_path = "filtered_form5500_combined.csv"
zip_output_path = "filtered_form5500_combined.zip"


# Then zip the CSV
with zipfile.ZipFile(zip_output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(csv_output_path)

print(f"Zipped CSV saved as: {zip_output_path}")
############################################################
import polars as pl
import os

# Folder containing your large CSVs
folder_path = "D:\PPD_data\Private_Investments"

# List all CSV files
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

# Read the column names of all files without loading full data
column_sets = []
for file in csv_files:
    df = pl.read_csv(file, n_rows=1)  # Only read header row
    column_sets.append(set(df.columns))

# Find common columns across all files
common_cols = set.intersection(*column_sets)

print(f"Common columns: {sorted(common_columns)}")


# Now read and append only common columns
for file in csv_files:
    try:
        df = pl.read_csv(file, n_rows=0)  # Read only header
        cols = set(df.columns)
        common_cols = cols if common_cols is None else common_cols & cols
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")


# Concatenate all DataFrames
combined_df = pl.concat(df_list, how="vertical")

print(f"Final shape: {combined_df.shape}")
##########################################################
import pandas as pd
import glob

# Settings
csv_folder = data_folder
csv_files = glob.glob(f"{csv_folder}/*.csv")
desired_columns = keywords  # columns you want in the final output

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    # Find columns present in both the file and the desired list
    present_cols = [col for col in desired_columns if col in df.columns]
    # Extract only those columns
    partial_df = df[present_cols].copy()
    dfs.append(partial_df)

# Concatenate, filling missing columns with NaN (pandas will do this)
concatenated = pd.concat(dfs, ignore_index=True)

# Reorder columns as in 'desired_columns', adding missing columns filled with NaN if not present in some files
for col in desired_columns:
    if col not in concatenated.columns:
        concatenated[col] = pd.NA
concatenated = concatenated[desired_columns]

# Save to CSV
concatenated.to_csv('concatenated.csv', index=False)
#############################################################
import pandas as pd
import glob

def process_files_recursively(csv_files, desired_columns, output_file, file_idx=0, first_chunk=True, chunksize=10_000):
    # Base case: Done with all files
    if file_idx == len(csv_files):
        return

    curr_file = csv_files[file_idx]
    for chunk in pd.read_csv(curr_file, chunksize=chunksize):
        # Keep only columns present in both chunk and desired list
        present_cols = [col for col in desired_columns if col in chunk.columns]
        partial_chunk = chunk[present_cols].copy()
        # Add missing columns as NaN
        for col in desired_columns:
            if col not in partial_chunk.columns:
                partial_chunk[col] = pd.NA
        # Reorder columns as per desired_columns
        partial_chunk = partial_chunk[desired_columns]
        # Write to output file
        partial_chunk.to_csv(
            output_file,
            mode='w' if first_chunk else 'a',
            index=False,
            header=first_chunk
        )
        first_chunk = False  # Only the first write has a header

    # Recurse for the next file
    process_files_recursively(csv_files, desired_columns, output_file, file_idx + 1, first_chunk, chunksize)

# --- Usage ---
csv_folder = data_folder
csv_files = glob.glob(f"{csv_folder}/*.csv")
desired_columns = keywords  # the columns you want
output_file = 'concatenated.csv'
process_files_recursively(csv_files, desired_columns, output_file)

concatenated = pd.read_csv(r"D:\concatenated.csv")
concatenated.columns