# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:49:26 2025

@author: 31908
"""
pip install groq

import pandas as pd
import re
from io import StringIO
from groq import Groq

# Initialize the Groq client with your API key
client = Groq(
    api_key="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"
)

# Example hospital name lists
hospital_list_1 = df1['SP_ENTITY_NAME'] 
#hospital_list_1  = df2['Company name Latin alphabet'] 
#hospital_list_1 = hospital_list_1[0:10000]

#hospital_list_2 = pd.read_csv(r"C:\Users\31908\Desktop\naicscode.csv")
#hospital_list_2 = pd.DataFrame(hospital_list_2.ENTITY_NAME.dropna())  
hospital_list_2 = df2['Company name Latin alphabet']

# Build the prompt for the LLM
prompt = f"""
You are a helpful assistant. Given two lists of hospital names, compare each item in List 1 to the closest matching name on the internet, even if there are slight differences in wording, abbreviations, or spelling. Provide the fuzzy matches in a clear csv table for the first 20 rows where you find 80% matches.
Please use sentence transformers for the task and also tell me how many tokens would you require for the matching along with printing the table for 20 rows.

List 1:
{hospital_list_1}

List 2:
{hospital_list_2}"""

# Make the API call
response = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ],
    stream=False
)

# Print the response
print(response.choices[0].message.content)

#################################################################
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

from huggingface_hub import login

# Login with your token
login(token="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")

# Initialize the model
model = SentenceTransformer('all-mpnet-base-v2')

# Load the data
df1 = CIQ_bk
df2 = orbis_bk

df2.nunique()

# Clean the data (convert to lowercase and strip whitespace)
df1['SP_ENTITY_NAME'] = df1['SP_ENTITY_NAME'].str.lower().str.strip()
df2['Company name Latin alphabet'] = df2['Company name Latin alphabet'].str.lower().str.strip()

# Encode the sentences
list1_embeddings = model.encode(str(df1['SP_ENTITY_NAME'].tolist()), show_progress_bar=True)
list2_embeddings = model.encode(str(df2['Company name Latin alphabet'].tolist()), show_progress_bar=True)

# Function to find closest matches
def find_closest_matches(emb1, list2_emb, list2_names, threshold=0.7):
    matches = []
    for emb in emb1:
        # Compute cosine similarity
        sims = np.dot(list2_emb, emb)
        sims = sims / (np.linalg.norm(list2_emb, axis=0) * np.linalg.norm(emb))
        # Find index of max similarity
        idx = np.argmax(sims)
        score = sims[idx]
        if score >= threshold:
            matches.append((list2_names[idx], score))
        else:
            matches.append((None, None))
    return matches

# Find matches for each item in list1
matches = find_closest_matches(list1_embeddings, list2_embeddings, df2['Company name Latin alphabet'].tolist())

# Create the result DataFrame
result = pd.DataFrame({
    'SP_ENTITY_NAME': df1['SP_ENTITY_NAME'][0:768].tolist(),
    'Closest_Entity_Name': [m[0] for m in matches],
    'Similarity_Score': [m[1] for m in matches]
})

# Export to CSV
result.to_csv('fuzzy_matches_m2.csv', index=False)
#############################################################################
#pip install spgci
import spgci as ci

ci.set_credentials('ashwath_pranjal@isb.edu', "C!5+4QhAB!UGcp!")
mdd = ci.MarketData()

mdd.get_symbols(commodity="Crude oil")

symbols = ["PCAAS00", "PCAAT00"]
mdd.get_assessments_by_symbol_current(symbol=symbols)
##########################################################################
#pip install faiss-cpu
import faiss

def normalize_name(name):
    return name.str.strip().lower()

df1['Normalized'] = df1['SP_ENTITY_NAME'].astype(str).str.strip().str.lower()
df2['Normalized'] = df2['Company name Latin alphabet'].astype(str).str.strip().str.lower()

list1 = df1['Normalized'].tolist()
list2 = df2['Normalized'].tolist()

# Create embeddings for both lists
embeddings_list1 = model.encode(list1, convert_to_tensor=True)
embeddings_list2 = model.encode(list2, convert_to_tensor=True)

# Convert embeddings to numpy arrays
embeddings_list1 = embeddings_list1.numpy()
embeddings_list2 = embeddings_list2.numpy()

# Convert to float32 for FAISS
embeddings_list2 = embeddings_list2.astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embeddings_list2.shape[1])
index.add(embeddings_list2)

# Parameters
k = 1  # Find the nearest neighbor
similarity_threshold = 0.8

# List to store results
results = []

# Search for each company in list1
for i, embedding in enumerate(embeddings_list1):
    embedding = embedding.reshape(1, -1).astype('float32')
    distances, indices = index.search(embedding, k)
    similarity = 1 - distances[0][0]  # Convert L2 distance to cosine similarity
    
    if similarity >= similarity_threshold:
        matched_company = list2[indices[0][0]]["ENTITY_NAME"]
        results.append({
            "Company_List1": list1[i]["Company"],
            "Matched_Company": matched_company,
            "Similarity": round(similarity, 2)
        })
        if len(results) >= 20:
            break

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results)

results_df.head()
results_df.to_csv("fuzzy_matches_CIQ_Orbis.csv", index=False)

ppd = pd.read_excel(r"D:\PPD_data\PensionInvestmentPerformanceDetailed.xlsx")

ppd_rtrn = ppd.loc[:,['PlanName'] + ppd.columns.str.contains("Rtrn")]
ppd_rtrn = ppd.loc[:, ['PlanName'] + ppd.columns[ppd.columns.str.contains("Rtrn")].tolist()]
ppd_rtrn.to_csv('PPD_Investment_Returns.csv',index = False)
# Extract columns containing 'Rtrn'
rtrn_columns = pd.DataFrame(ppd.columns[ppd.columns.str.contains("Rtrn")])

rtrn_columns = rtrn_columns.rename(columns = {0:'varname'})

#rtrn_columns.to_csv('Returns.csv')
# Drop them from the original DataFrame
ppd = ppd.drop(columns=rtrn_columns[0])

cols = ppd.columns
rtrn_columns = rtrn_columns["varname"].str.replace("_all", "", case=False).str.strip()
rtrn_columns = pd.DataFrame(rtrn_columns)

codebook = pd.read_excel(r"D:\Orbis\Active\Investment-Codebook.xlsx")
codebook.columns
ppd_inv_returns = rtrn_columns.merge(codebook, on = 'varname', how = 'left')
ppd_inv_returns.to_csv('variable_description.csv',index = False)
#########################################################################
import os
import requests
import zipfile
import pyreadstat
import pandas as pd

os.chdir(r"D:\PPD_data\Private_Investments")


# Base URL pattern

base_url_pattern = "https://askebsa.dol.gov/opr/{year}-5000-CD-ROM.zip"

# Output directory
base_extract_dir = "all_extracted_sas_files"
os.makedirs(base_extract_dir, exist_ok=True)

# Dictionary to hold year-wise combined DataFrames
combined_by_year = {}

# Loop through years
for year in [2015, 2023]:
    print(f"\nProcessing year: {year}")
    zip_url = base_url_pattern.format(year=year)
    zip_filename = f"pension_research_{year}.zip"
    zip_path = os.path.join(base_extract_dir, zip_filename)

    try:
        # Download the ZIP file
        response = requests.get(zip_url)
        if response.status_code != 200:
            print(f"Failed to download {zip_url} (status code: {response.status_code})")
            continue

        with open(zip_path, 'wb') as f:
            f.write(response.content)

        print(f"Downloaded ZIP for {year}")

        # Extract ZIP file
        extract_dir = os.path.join(base_extract_dir, str(year))
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"Extracted files for {year}")

        # Collect all yearly DataFrames to append
        yearly_dfs = []

        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith('.sas7bdat'):
                    file_path = os.path.join(root, file)
                    print(f"Reading {file}...")
                    try:
                        df, meta = pyreadstat.read_sas7bdat(file_path)
                        df["source_file"] = file
                        yearly_dfs.append(df)
                    except Exception as e:
                        print(f"Failed to read {file}: {e}")

        if yearly_dfs:
            combined_df = pd.concat(yearly_dfs, ignore_index=True, sort=False)
            combined_by_year[year] = combined_df
            print(f"Appended {len(yearly_dfs)} files for year {year} into a single DataFrame")

    except Exception as e:
        print(f"Error processing year {year}: {e}")

# Optional: Combine all years into one master DataFrame
all_years_combined = pd.concat(combined_by_year.values(), ignore_index=True, sort=False)
print(f"\n✅ Final combined dataset has {len(all_years_combined)} rows from {len(combined_by_year)} years.")

all_years_combined.to_csv('Private_Pension_Plans4.csv', index=False)
#############################################################
# Path to the folder containing the CSV files
#os.chdir("D:\PPD_data")
import os
import pandas as pd

folder_path = "D:\PPD_data\Private_Investments"
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

chunk_size = 50000  # Adjust based on your memory limits
all_chunks = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    print(f"Reading {file} in chunks...")
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        all_chunks.append(chunk)

# Concatenate all chunks into one DataFrame
combined_df = pd.concat(all_chunks, ignore_index=True)

print(f"✅ Combined DataFrame shape: {combined_df.shape}")


# Preview the result
print(f"Combined {len(csv_files)} CSV files into a DataFrame with {len(all_csv_data)} rows.")
df = pd.read_csv(r"D:\PPD_data\Private_Investments\Private_Pension_Plans4.csv")
cols = pd.DataFrame(df.columns)

cols = cols.rename(columns = {0:'Variables'})

os.chdir("D:\PPD_data")


cols.to_csv('pvt_ins.csv',index =False)
###################################
os.chdir(r"D:\PPD_data")
import os
import csv
import glob
from pathlib import Path


def append_large_csv_files(folder_path, output_file, chunk_size=10000, include_headers=True):
    """
    Append extremely large CSV files with minimum space complexity.
    
    Args:
        folder_path (str): Path to folder containing CSV files
        output_file (str): Path for the output combined CSV file
        chunk_size (int): Number of rows to process at once (adjust based on available memory)
        include_headers (bool): Whether to include headers from first file only
    """
    folder_path = r"D:\PPD_data\Private_Investments"
    csv_files = list(folder_path.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Sort files for consistent processing order
    csv_files.sort()
    
    first_file = True
    total_rows_written = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = None
        
        for i, csv_file in enumerate(csv_files):
            print(f"Processing file {i+1}/{len(csv_files)}: {csv_file.name}")
            
            try:
                with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    
                    # Handle headers
                    try:
                        headers = next(reader)
                        if first_file and include_headers:
                            writer = csv.writer(outfile)
                            writer.writerow(headers)
                            total_rows_written += 1
                        elif first_file:
                            writer = csv.writer(outfile)
                        # Skip headers in subsequent files if include_headers is True
                        
                        first_file = False
                    except StopIteration:
                        print(f"Warning: {csv_file.name} is empty, skipping...")
                        continue
                    
                    # Process file in chunks
                    chunk = []
                    rows_processed = 0
                    
                    for row in reader:
                        chunk.append(row)
                        rows_processed += 1
                        
                        # Write chunk when it reaches the specified size
                        if len(chunk) >= chunk_size:
                            writer.writerows(chunk)
                            total_rows_written += len(chunk)
                            chunk.clear()  # Clear chunk to free memory
                            
                            if rows_processed % (chunk_size * 10) == 0:
                                print(f"  Processed {rows_processed:,} rows from {csv_file.name}")
                    
                    # Write remaining rows in chunk
                    if chunk:
                        writer.writerows(chunk)
                        total_rows_written += len(chunk)
                        chunk.clear()
                    
                    print(f"  Completed {csv_file.name}: {rows_processed:,} rows")
                    
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                continue
    
    print(f"\nAppending completed!")
    print(f"Total rows written: {total_rows_written:,}")
    print(f"Output file: {output_file}")
    print(f"Output file size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
    
import pandas as pd
cols.columns
# Load the CSV file
cols = pd.read_csv(r"D:\PPD_data\pvt_ins.csv", encoding = 'windows-1252')

# Add comma at the end of each value in the 'variable' column
cols['SAS Variable Name '] = cols['SAS Variable Name '].astype(str) + ','

cols = cols.dropna(subset = ['SAS Variable Name '])

cols[cols['SAS Variable Name '] == nan].nunique()
cols.columns
cols['Variable Format '].unique()

cols = cols.drop(columns = ['SQL_type'])

def infer_type(column):
    if column == 'Character ':
        return 'TEXT'
    elif column == 'Numeric ':
        return 'INT'
    else:
        return 'nan'

cols['SQL_type'] = cols['Variable Format '].apply(infer_type)
# Save the modified DataFrame back to CSV
cols.to_csv("pvt_ins_modified.csv", index=False)
#################################################
#Multiprocessing with Pandas
df = pd.read_csv(r"D:\PPD_data\Private_Investments\Private_Pension_Plans.csv")
df2 = pd.read_csv(r"D:\PPD_data\Private_Investments\Private_Pension_Plans2.csv")
df3 = pd.read_csv(r"D:\PPD_data\Private_Investments\Private_Pension_Plans3.csv")
df4 = pd.read_csv(r"D:\PPD_data\Private_Investments\Private_Pension_Plans4.csv")

import pandas as pd
import os

file_path = "D:\PPD_data\Private_Investments\Private_Pension_Plans.csv"

def split_csv(file_path, output_dir="output_chunks", chunk_size=100000, output_format="csv"):
    os.makedirs(output_dir, exist_ok=True)
    file_basename = os.path.splitext(os.path.basename(file_path))[0]

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        output_file = os.path.join(output_dir, f"{file_basename}_part{i+1}.{output_format}")
        
        if output_format == "csv":
            chunk.to_csv(output_file, index=False)
        elif output_format == "parquet":
            chunk.to_parquet(output_file, index=False)
        elif output_format == "feather":
            chunk.to_feather(output_file)
        else:
            raise ValueError("Unsupported format. Use 'csv', 'parquet', or 'feather'.")
        
        print(f"Saved: {output_file}")

# 🔧 Example usage
split_csv("your_file.csv", chunk_size=50000, output_format="parquet")
#################################################################################
#all sas files into one
import os
import pandas as pd
import pyreadstat

extract_dir = r"D:\PPD_data\Private_Investments\all_extracted_sas_files"
all_dfs = []

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.lower().endswith('.sas7bdat'):
            file_path = os.path.join(root, file)
            print(f"Reading {file_path}...")
            try:
                df, meta = pyreadstat.read_sas7bdat(file_path)
                # Keep only rows where DB == 1
                if 'DB' in df.columns:
                    df = df[df['DB'] == 1]
                else:
                    print(f"Column 'DB' not found in {file}, skipping this file.")
                    continue
                df["source_file"] = file
                # Extract the "year" or parent folder of subfolder
                parent_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                df["year_folder"] = parent_folder
                all_dfs.append(df)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

if all_dfs:
    master_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    print(f"\n✅ Final combined dataset has {len(master_df)} rows from {len(all_dfs)} files.")
else:
    print("No SAS datasets with DB==1 were found or loaded.")

All_DB = master_df

All_DB = All_DB.drop_duplicates()
##################################################################################################################
