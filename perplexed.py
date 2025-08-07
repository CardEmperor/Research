# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:04:49 2025

@author: 31908
"""

import pandas as pd
import glob

# Define keywords for identifying, location, and investment variables
keywords = [
    "ACK_ID", 'FILING_ID', "PLAN_YR", 'DB', "PLAN_YEAR","FY_BGN_DT","FY_END_DT","DT_RECEIVED","DT_PLAN_EFFECTIVE","DT_SIGNED","PROC_DT", "PROC_DATE","FILING_DATE","PLAN_EFFECTIVE_DATE"
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

# Define the folder containing your CSV files
data_folder = "D:/PPD_data/Private_Investments"


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
desired_columns = key_cols  # the columns you want
output_file = 'concatenated.csv'
process_files_recursively(csv_files, desired_columns, output_file)

########################################################
import pandas as pd
import glob

def process_files_recursively(csv_files, desired_columns, output_file, db_filter_value,
                              file_idx=0, first_chunk=True, chunksize=10_000, db_column="DB"):
    # Base case: All files processed
    if file_idx == len(csv_files):
        return

    curr_file = csv_files[file_idx]
    for chunk in pd.read_csv(curr_file, chunksize=chunksize):
        # Filter rows by DB column value
        # Only consider rows where DB column exists and is equal to db_filter_value
        if db_column in chunk.columns:
            chunk = chunk[chunk[db_column] == db_filter_value]
        else:
            # If DB column is missing from file, skip all rows for that chunk
            chunk = chunk.iloc[0:0]

        # Keep only columns that are both present in chunk and desired
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
    process_files_recursively(csv_files, desired_columns, output_file, db_filter_value,
                             file_idx + 1, first_chunk, chunksize, db_column)

# --- Usage ---
csv_folder = data_folder
csv_files = glob.glob(f"{csv_folder}/*.csv")
desired_columns = key_cols  # include 'DB' if you want it in output
output_file = 'concatenated_DB.csv'
db_filter_value = 1  # set your DB column value to filter
process_files_recursively(csv_files, desired_columns, output_file, db_filter_value)
############################################################################
import pandas as pd
import glob

def process_files_recursively(csv_files, output_file, db_filter_value, file_idx=0, first_chunk=True, chunksize=10_000, db_column="DB"):
    # Base case: All files processed
    if file_idx == len(csv_files):
        return

    curr_file = csv_files[file_idx]
    for chunk in pd.read_csv(curr_file, chunksize=chunksize):
        # Filter rows where DB == db_filter_value
        if db_column in chunk.columns:
            filtered_chunk = chunk[chunk[db_column] == db_filter_value]
        else:
            # If DB column is missing, skip all rows for this chunk
            filtered_chunk = chunk.iloc[0:0]
        # Write to output file (write header only for first chunk)
        filtered_chunk.to_csv(
            output_file,
            mode='w' if first_chunk else 'a',
            index=False,
            header=first_chunk
        )
        first_chunk = False  # Only the first write has header

    # Recurse for the next file
    process_files_recursively(csv_files, output_file, db_filter_value, file_idx + 1, first_chunk, chunksize, db_column)

# --- Usage ---
csv_folder = data_folder
csv_files = glob.glob(f"{csv_folder}/*.csv")
output_file = 'db_concatenated_all_years.csv'
db_filter_value = 1  # Filter for DB == 1

process_files_recursively(csv_files, output_file, db_filter_value)
#######################################################################
import pandas as pd
all_years_db = pd.read_csv(r"D:\concatenated_DB.csv")
all_years_db = all_years_db.drop_duplicates()
    
All_DB['FILING_ID'] = All_DB['FILING_ID'].fillna(All_DB['ACK_ID'])

all_years_db = All_DB.dropna(axis=1,how='all')

cols = all_years_db.columns
key_cols = ['FILING_ID','PYB','PYE','PLAN_NAME','OPR_EIN','OPR_PN','SPONSOR_DFE_NAME','SPONS_DFE_CITY','SPONS_DFE_STATE','SPONS_DFE_ZIP_CODE','TYPE_PLAN_FILING_IND',
            'TOT_PARTCP_BOY_CNT','TOT_ACTIVE_PARTCP_CNT','ACQUIS_INDBT_EOY_AMT','AGGREGATE_PROCEEDS_AMT','AGGREGATE_COSTS_AMT',
            'COMMON_STOCK_EOY_AMT','CONTRACT_ADMIN_FEES_AMT','CORP_DEBT_OTHER_EOY_AMT','CORP_DEBT_PREFERRED_EOY_AMT','INS_CARRIER_BNFTS_AMT', 
            'INS_CO_GEN_ACCT_EOY_AMT', 'INT_103_12_INVST_EOY_AMT', 'INT_BEAR_CASH_AMT', 'INT_BEAR_CASH_EOY_AMT', 'INT_COMMON_TR_EOY_AMT', 
            'INT_MASTER_TR_EOY_AMT', 'INT_ON_CORP_DEBT_AMT', 'INT_ON_GOVT_SEC_AMT', 'INT_ON_OTH_INVST_AMT', 'INT_ON_OTH_LOANS_AMT', 
            'INT_ON_PARTCP_LOANS_AMT', 'INT_POOL_SEP_ACCT_EOY_AMT', 'INT_REG_INVST_CO_EOY_AMT','INVST_MGMT_FEES_AMT','JOINT_VENTURE_EOY_AMT', 
            'NET_ASSETS_BOY_AMT','NET_ASSETS_EOY_AMT','NET_INCOME_AMT', 'NON_CASH_CONTRIB_BS_AMT', 'NON_INT_BEAR_CASH_EOY_AMT', 'OPRTNG_PAYABLE_EOY_AMT', 
            'OTH_BNFT_PAYMENT_AMT', 'OTH_CONTRIB_RCVD_AMT', 'OTH_INVST_EOY_AMT', 'OTHER_ADMIN_FEES_AMT', 'OTHER_LOANS_EOY_AMT', 'OTHER_RECEIVABLES_BOY_AMT',
            'OTHER_RECEIVABLES_EOY_AMT', 'PARTCP_CONTRIB_BOY_AMT', 'PARTCP_CONTRIB_EOY_AMT', 'PARTCP_LOANS_EOY_AMT', 'PARTICIPANT_CONTRIB_AMT', 'PREF_STOCK_EOY_AMT',
            'PROFESSIONAL_FEES_AMT', 'REAL_ESTATE_EOY_AMT', 'TOT_ADMIN_EXPENSES_AMT', 'TOT_ASSETS_EOY_AMT', 'TOT_CONTRIB_AMT', 'TOT_CORRECTIVE_DISTRIB_AMT', 'TOT_DISTRIB_BNFT_AMT',
            'TOT_DM_DISTRIB_PTCP_LNS_A', 'TOT_EXPENSES_AMT', 'TOT_GAIN_LOSS_SALE_AST_AMT', 'TOT_INCOME_AMT','OTHER_INCOME_AMT','TOT_INT_EXPENSE_AMT','TOT_LIABILITIES_EOY_AMT',
            'TOT_TRANSFERS_FROM_AMT', 'TOT_TRANSFERS_TO_AMT','TOT_UNREALZD_APPRCTN_AMT','TOTAL_DIVIDENDS_AMT','TOTAL_INTEREST_AMT','TOTAL_RENTS_AMT','UNREALZD_APPRCTN_OTH_AMT',
            'UNREALZD_APPRCTN_RE_AMT','NON_INT_BEAR_CASH_BOY_AMT','INT_BEAR_CASH_BOY_AMT','GOVT_SEC_BOY_AMT','CORP_DEBT_PREFERRED_BOY_AMT', 'CORP_DEBT_OTHER_BOY_AMT', 'PREF_STOCK_BOY_AMT',
            'COMMON_STOCK_BOY_AMT',	'JOINT_VENTURE_BOY_AMT','ASSET_UNDETERM_VAL_IND','ASSET_UNDETERM_VAL_AMT','NON_CASH_CONTRIB_IND','NON_CASH_CONTRIB_AMT','AST_HELD_INVST_IND','FIVE_PRCNT_TRANS_IND',
            'RES_TERM_PLAN_ADPT_AMT','FAIL_PROVIDE_BENEFIT_DUE_IND','FAIL_PROVIDE_BENEFIT_DUE_AMT','PLAN_BLACKOUT_PERIOD_IND','COMPLY_BLACKOUT_NOTICE_IND','SCH_I_PLAN_YEAR_BEGIN_DATE','SCH_I_TAX_PRD','SCH_I_PLAN_NUM',
            'SCH_I_EIN','SMALL_DEEM_DSTRB_PARTCP_LN_AMT','SMALL_JOINT_VENTURE_EOY_IND','SMALL_EMPLR_PROP_EOY_IND','SMALL_INV_REAL_ESTATE_EOY_IND','SMALL_INV_REAL_ESTATE_EOY_AMT','SF_TOT_ASSETS_EOY_AMT','SF_TOT_LIABILITIES_EOY_AMT','SF_NET_ASSETS_EOY_AMT','SF_NET_ASSETS_BOY_AMT','SF_TOT_LIABILITIES_BOY_AMT','SF_TOT_ASSETS_BOY_AMT','SF_EMPLR_CONTRIB_INCOME_AMT','SF_PARTICIP_CONTRIB_INCOME_AMT','SF_OTH_CONTRIB_RCVD_AMT','SF_OTHER_INCOME_AMT','SF_TOT_INCOME_AMT','SF_TOT_DISTRIB_BNFT_AMT','SF_CORRECTIVE_DEEMED_DISTR_AMT','SF_ADMIN_SRVC_PROVIDERS_AMT','SF_OTH_EXPENSES_AMT','SF_TOT_EXPENSES_AMT','SF_NET_INCOME_AMT',
            'SF_TOT_PLAN_TRANSFERS_AMT',	'SF_TOT_ASSETS_BOY_AME','SF_EMPLR_CONTRIB_INCOME_AME','SF_PARTICIP_CONTRIB_INCOME_AME','SF_OTH_CONTRIB_RCVD_AME','SF_TOT_DISTRIB_BNFT_AME','SF_ADMIN_SRVC_PROVIDERS_AME','SMALL_ADMIN_SRVC_PROVIDERS_AME','REGISTERED_INVST_AME','TOT_DEEMED_DISTR_PART_LNS_AME','SF_CORRECTIVE_DEEMED_DISTR_AME','SF_NET_ASSETS_BOY_AME','SF_NET_ASSETS_EOY_AME','SF_NET_INCOME_AME',
            'SF_OTHER_INCOME_AME','SF_OTH_EXPENSES_AME','SF_TOT_ASSETS_EOY_AME','SF_TOT_EXPENSES_AME','SF_TOT_INCOME_AME','SF_TOT_LIABILITIES_BOY_AME','SF_TOT_LIABILITIES_EOY_AME','SF_TOT_PLAN_TRANSFERS_AME','TYPE_PLAN_ENTITY_CD_E','PLAN_NAME_E','funding_arrange','benefit_arrange','TRANSFERS_CHANGE','ERRORCHECK','non_cash_contr_bs_AME','UNSPEC_INVEST_EOY_AME','OTHER_EXPENSES_AME','TOT_ASSETS_BOY_ame','TOT_LIABILITIES_BOY_ame','UNSPEC_DIST_BNFT_AME','TOT_ADMIN_SRVC_PROVIDERS_AME','TOT_PARTCP_BOY_AME','CORRECTIVE_DEEMED_DISTR_AME'
]

all_years_db_key_cols = all_years_db[key_cols]

all_years_db_key_cols = all_years_db_key_cols.drop_duplicates()

# Drop columns with more than 50% missing values
threshold = 0.5
all_years_db_key_cols = all_years_db_key_cols.loc[:, all_years_db_key_cols.isnull().mean() < threshold]

all_years_db_key_cols.isnull().sum()

# Keep only the first occurrence of each PLAN_NAME
all_years_db_unique = all_years_db_key_cols.drop_duplicates(subset='PLAN_NAME', keep='first')

# Optional: Save result
#all_years_db_unique.to_csv('filtered_plan_names.csv', index=False)

#all_years_db_key_cols.to_csv('concatenated_DB_keycols.csv',index=False)

all_years_db_key_cols.PYB.isnull().sum()

pyreadstat.write_xport(all_years_db_key_cols, "private_plans_key_cols.xpt")

#pyreadstat.write_xport(all_years_db_unique, "unique_plans.xpt")
#cols = pd.DataFrame(cols).rename(columns={0:'Columns'})
################################################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\31908\Dropbox\RA Share File 2024-25\ppd-data-latest.csv",encoding='ISO-8859-1')


# Filter data from year 2000 onward
df_filtered = df[df['fy'] >= 2000].copy()
df_filtered['underfunding'] = df['ActAssets_GASB']/df['ActLiabilities_GASB']

# Define columns
fixed_income_col = 'FITotal_Actl'
equities_col = 'EQTotal_Actl'
underfunding_col = 'underfunding'

# Define alternative columns (excluding the above)
all_columns = ['FITotal_Actl', 'EQTotal_Actl', 'PETotal_Actl', 'RETotal_Actl', 
               'COMDTotal_Actl', 'HFTotal_Actl', 'AltMiscTotal_Actl', 'underfunding']
alternative_cols = [col for col in all_columns if col not in [fixed_income_col, equities_col, underfunding_col]]

# Create new dataframe with required structure
df_filtered['Alternatives'] = df_filtered[alternative_cols].sum(axis=1)

# Prepare the new dataframe with just the four categories
df_four = df_filtered[['fy', 'underfunding']]

# Group by year and compute mean
avg_by_year = df_four.groupby('fy').mean()

avg_underfunding = df_filtered.groupby
# Rename columns for clarity in plot
avg_by_year.rename(columns={
    fixed_income_col: 'Fixed Income',
    equities_col: 'Equities'
#    underfunding_col: 'Underfunding'
}, inplace=True)

# Plot
ax = avg_by_year.plot(marker='o', figsize=(10, 6))

plt.title("(A) Public Pension Plans Underfunding")
plt.xlabel("Year")
plt.ylabel("Allocation (%)")
plt.grid(True)

# Move legend outside
ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend

plt.show()

core_cols = ['FITotal_Actl', 'EQTotal_Actl', 'underfunding']

# All other columns are considered alternatives
alternative_cols = [col for col in all_columns if col not in core_cols + ['fy']]

# Filter from year 2000 onwards
df_2000_onward = df[df['fy'] >= 2000]
df_2000_onward.fy = df_2000_onward.fy.astype(int)

# Group and average alternative categories by year
alt_avg_by_year = df_2000_onward.groupby('fy')[alternative_cols].mean()

# Rename columns for clarity in plot
alt_avg_by_year.rename(columns={
    'PETotal_Actl': 'Private Equity',
    'RETotal_Actl': 'Real Estate',
   'COMDTotal_Actl': 'Commodities',
   'HFTotal_Actl': 'Hedge Funds',
   'AltMiscTotal_Actl': 'Alternatives Misc.'
}, inplace=True)

# Plot stacked vertical bar chart
ax = alt_avg_by_year.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20c')

plt.title("(B) Within alternative assets")
plt.xlabel("Year")
plt.ylabel("Allocation (%)")
plt.legend(title="Alternative Categories", bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.show()




































