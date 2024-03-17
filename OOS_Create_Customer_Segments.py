#%% Import packages
import pandas as pd


#%% Load the dataset
file_path = '/transaction_data.parquet'
df1 = pd.read_parquet(file_path)


#%% Filter the dataset for the desired date range and drop irrelevant variables
df = df1[(df1['transaction_date'] >= '2019-01-01') & (df1['transaction_date'] < '2024-01-01')]
df.drop(columns=['account_type', 'transaction_date', 'transaction_number', 'tax_payment_yn',
                 'same_portfolio_transaction_yn', 'all_same_people_transaction_yn',
                 'some_same_people_transaction_yn', 'transaction_year'], inplace=True)
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp']).dt.floor('S')


#%% Create subsets for retail and business transactions
retail_df = df[df['business_or_retail'] == 'Retail']
business_df = df[df['business_or_retail'] == 'Business']


#%% Function to apply censoring logic
def apply_censoring(dataframe):
    dataframe.loc[(dataframe['balance_before'] >= 0) & (dataframe['balance_after'] < 0), 'balance_after'] = 0
    dataframe.loc[(dataframe['balance_before'] < 0) & (dataframe['balance_after'] >= 0), 'balance_before'] = 0
    dataframe = dataframe[~((dataframe['balance_before'] <= 0) & (dataframe['balance_after'] <= 0))]
    return dataframe


#%% Apply censoring
retail_censored = apply_censoring(retail_df).copy()
business_censored = apply_censoring(business_df).copy()


#%% Segment data based on in-sample segments
# Define file paths for the customer ID files
file_paths = {
    'retail_1': '/Users/hhsmid/Desktop/Hugo EUR/09. Seminar Financial Case Studies/03. Data/Final data/sampled_ids/R1_censored_sampled_ids_long.xlsx',
    'retail_2': '/Users/hhsmid/Desktop/Hugo EUR/09. Seminar Financial Case Studies/03. Data/Final data/sampled_ids/R2_censored_sampled_ids_long.xlsx',
    'retail_3': '/Users/hhsmid/Desktop/Hugo EUR/09. Seminar Financial Case Studies/03. Data/Final data/sampled_ids/R3_censored_sampled_ids_long.xlsx',
    'business_1': '/Users/hhsmid/Desktop/Hugo EUR/09. Seminar Financial Case Studies/03. Data/Final data/sampled_ids/B1_censored_sampled_ids_long.xlsx',
    'business_2': '/Users/hhsmid/Desktop/Hugo EUR/09. Seminar Financial Case Studies/03. Data/Final data/sampled_ids/B2_censored_sampled_ids_long.xlsx',
    'business_3': '/Users/hhsmid/Desktop/Hugo EUR/09. Seminar Financial Case Studies/03. Data/Final data/sampled_ids/B3_censored_sampled_ids_long.xlsx'
}

# Load the customer IDs into dictionaries
customer_segments = {}
for segment, path in file_paths.items():
    customer_segments[segment] = pd.read_excel(path)['id_rand'].unique()

# Tag the transaction data with the customer segment
def tag_customer_segment(row):
    for segment, ids in customer_segments.items():
        if row['id_rand'] in ids:
            return segment
    return 'Unknown'

retail_censored['customer_segment'] = retail_censored.apply(tag_customer_segment, axis=1)
business_censored['customer_segment'] = business_censored.apply(tag_customer_segment, axis=1)

# Retail segments
retail_segment_1 = retail_censored[retail_censored['customer_segment'] == 'retail_1']
retail_segment_2 = retail_censored[retail_censored['customer_segment'] == 'retail_2']
retail_segment_3 = retail_censored[retail_censored['customer_segment'] == 'retail_3']

# Business segments
business_segment_1 = business_censored[business_censored['customer_segment'] == 'business_1']
business_segment_2 = business_censored[business_censored['customer_segment'] == 'business_2']
business_segment_3 = business_censored[business_censored['customer_segment'] == 'business_3']


#%% Safe to parquet files
base_dir = 'OOS customer segmented data/'

# Retail segments
retail_segment_1.to_parquet(f"{base_dir}oos_retail_segment_1_long.parquet")
retail_segment_2.to_parquet(f"{base_dir}oos_retail_segment_2_long.parquet")
retail_segment_3.to_parquet(f"{base_dir}oos_retail_segment_3_long.parquet")

# Business segments
business_segment_1.to_parquet(f"{base_dir}oos_business_segment_1_long.parquet")
business_segment_2.to_parquet(f"{base_dir}oos_business_segment_2_long.parquet")
business_segment_3.to_parquet(f"{base_dir}oos_business_segment_3_long.parquet")

