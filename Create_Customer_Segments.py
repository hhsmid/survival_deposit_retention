#%% Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jenkspy


#%% Load the dataset
file_path = ''
df1 = pd.read_parquet(file_path)


#%% Filter the dataset for the desired date range and drop irrelevant variables
df = df1[(df1['transaction_date'] >= '2014-01-01') & (df1['transaction_date'] < '2019-01-01')]
df.drop(columns=['account_type', 'transaction_date', 'transaction_number', 'tax_payment_yn',
                 'same_portfolio_transaction_yn', 'all_same_people_transaction_yn',
                 'some_same_people_transaction_yn', 'transaction_year'], inplace=True)
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp']).dt.floor('S')


#%% Create subsets for retail and business transactions
retail_df = df[df['business_or_retail'] == 'Retail']
business_df = df[df['business_or_retail'] == 'Business']


#%% Descriptive statistics and boxplots for balance_after
retail_balance_descriptive_stats = retail_df['balance_after'].describe()
business_balance_descriptive_stats = business_df['balance_after'].describe()

print("Retail Customer Balances Descriptive Statistics:\n", retail_balance_descriptive_stats)
print("\n Business Customer Balances Descriptive Statistics: \n", business_balance_descriptive_stats)

# Box-plot for Retail Customer Balances
plt.figure(figsize=(10, 6))  # Set figure size for better readability
sns.boxplot(x=retail_df['balance_after'])
plt.title('Boxplot of Retail Customer Balances')
plt.xlabel('Balance')
plt.show()

# Box-plot for Business Customer Balances
plt.figure(figsize=(10, 6))  # Set figure size for better readability
sns.boxplot(x=business_df['balance_after'])
plt.title('Boxplot of Business Customer Balances')
plt.xlabel('Balance')
plt.show()

# Most transactions had a positive balance_after, some had a negative balance_after and distribution of balance_after is significantly right skewed.
# Both for retail and business, outliers in balance_after are mainly in the right tail.
# Retail has one clear outlier in the right tail.


#%% Count and summarize negative balance_after observations
# Retail Customers
negative_retail_balances = retail_df[retail_df['balance_after'] < 0]
count_negative_retail = negative_retail_balances.shape[0]

unique_negative_retail_ids = negative_retail_balances['id_rand'].unique()
count_unique_negative_retail_ids = unique_negative_retail_ids.shape[0]

print(f"Number of Negative Balance Observations in Retail: {count_negative_retail}")
print(f"Number of Unique Retail Customer IDs with Negative Balances: {count_unique_negative_retail_ids}")


# Business Customers
negative_business_balances = business_df[business_df['balance_after'] < 0]
count_negative_business = negative_business_balances.shape[0]

unique_negative_business_ids = negative_business_balances['id_rand'].unique()
count_unique_negative_business_ids = unique_negative_business_ids.shape[0]

print(f"Number of Negative Balance Observations in Business: {count_negative_business}")
print(f"Number of Unique Business Customer IDs with Negative Balances: {count_unique_negative_business_ids}")

# For the majority of the transactions with a negative balance_after, the transaction was a withdrawal (negative amount). 
# However, in some extreme cases, a positive transaction took place for an account with a balance_before already negative.


#%% Function to apply censoring logic
def apply_censoring(dataframe):
    dataframe.loc[(dataframe['balance_before'] >= 0) & (dataframe['balance_after'] < 0), 'balance_after'] = 0
    dataframe.loc[(dataframe['balance_before'] < 0) & (dataframe['balance_after'] >= 0), 'balance_before'] = 0
    dataframe = dataframe[~((dataframe['balance_before'] <= 0) & (dataframe['balance_after'] <= 0))]
    return dataframe


#%% Apply censoring
retail_censored = apply_censoring(retail_df)
business_censored = apply_censoring(business_df)


#%% Descriptive statistics for balance_after after censoring
retail_balance_censored_descriptive_stats = retail_censored['balance_after'].describe()
business_balance_censored_descriptive_stats = business_censored['balance_after'].describe()

print("Censored Retail Customer Balances Descriptive Statistics:\n", retail_balance_censored_descriptive_stats)
print("Censored Business Customer Balances Descriptive Statistics:\n", business_balance_censored_descriptive_stats)


#%% Function to calculate time-weighted average balance for each account
def calculate_time_weighted_average_balance(dataframe):
    # Define the start and end of the analysis period
    start_analysis_period = pd.to_datetime('2014-01-01')
    end_analysis_period = pd.to_datetime('2019-01-01')
    
    dataframe = dataframe.sort_values(by=['id_rand', 'transaction_timestamp'])
    
    # Calculate the duration from the start of the analysis period to the first transaction
    dataframe['previous_transaction_timestamp'] = dataframe.groupby('id_rand')['transaction_timestamp'].shift(1)
    first_transaction_mask = dataframe['previous_transaction_timestamp'].isna()
    dataframe.loc[first_transaction_mask, 'previous_transaction_timestamp'] = start_analysis_period
    
    # Calculate the duration to the next transaction or the end of the analysis period
    dataframe['next_transaction_timestamp'] = dataframe.groupby('id_rand')['transaction_timestamp'].shift(-1)
    dataframe.loc[dataframe['next_transaction_timestamp'].isna(), 'next_transaction_timestamp'] = end_analysis_period
    
    # Calculate durations
    dataframe['duration'] = (dataframe['next_transaction_timestamp'] - dataframe['transaction_timestamp']).dt.total_seconds()
    dataframe.loc[first_transaction_mask, 'duration'] = (dataframe['transaction_timestamp'] - start_analysis_period).dt.total_seconds()
    
    # Calculate the time-weighted balance, adjusting for the initial balance before the first transaction
    dataframe['time_weighted_balance'] = dataframe['balance_after'] * dataframe['duration']
    dataframe.loc[first_transaction_mask, 'time_weighted_balance'] = dataframe['balance_before'] * dataframe.loc[first_transaction_mask, 'duration']
    
    # Aggregate the time-weighted balance and total duration for each account
    customer_balances = dataframe.groupby('id_rand').agg(
        total_time_weighted_balance=('time_weighted_balance', 'sum'),
        total_duration=('duration', 'sum')
    ).reset_index()
    
    # Calculate the time-weighted average balance for each account
    customer_balances['time_weighted_average_balance'] = customer_balances['total_time_weighted_balance'] / customer_balances['total_duration']
    
    return customer_balances


#%% Calculate time-weighted average balance for accounts in retail and business
retail_accounts = calculate_time_weighted_average_balance(retail_censored)
business_accounts = calculate_time_weighted_average_balance(business_censored)


#%% Descriptive statistics and boxplots for the time-weighted average balance
retail_av_balance_descriptive_stats = retail_accounts['time_weighted_average_balance'].describe()
business_av_balance_descriptive_stats = business_accounts['time_weighted_average_balance'].describe()

print("Retail Customer Time-Weighted Average Balance Descriptive Statistics:\n", retail_av_balance_descriptive_stats)
print("Business Customer Time-Weighted Average Balance Descriptive Statistics:\n", business_av_balance_descriptive_stats)

# Box-plot for Retail Customer Time-Weighted Average Balance
plt.figure(figsize=(10, 6))  # Set figure size for better readability
sns.boxplot(x=retail_accounts['time_weighted_average_balance'])
plt.title('Boxplot of Retail Customer Time-Weighted Average Balances')
plt.xlabel('Time-Weighted Average Balance')
plt.show()

# Box-plot for Business Customer Time-Weighted Average Balance
plt.figure(figsize=(10, 6))  # Set figure size for better readability
sns.boxplot(x=business_accounts['time_weighted_average_balance'])
plt.title('Boxplot of Business Customer Time-Weighted Average Balances')
plt.xlabel('Time-Weighted Average Balance')
plt.show()

# Business segment has slightly higher mean time-weighted average balance.
# Time-weighted average balance is mostly positive for both segments.
# For some accounts time-weighted average balance is negative for both segments.


#%% Deleting outliers
# Retail
# Step 1: Find the id_rand values with time_weighted_average_balance > 800,000 based on boxplots
retail_ids_to_remove = retail_accounts[retail_accounts['time_weighted_average_balance'] > 800000]['id_rand'].unique()

# Step 2: Remove rows with those id_rand values from the DataFrame
retail_accounts_clean = retail_accounts[~retail_accounts['id_rand'].isin(retail_ids_to_remove)]
retail_censored_cleaned = pd.merge(retail_censored, retail_accounts_clean[['id_rand']], on='id_rand', how='inner')

# Business
# Step 1: Find the id_rand values with time_weighted_average_balance > 1,500,000 based on boxplots
business_ids_to_remove = business_accounts[business_accounts['time_weighted_average_balance'] > 1000000]['id_rand'].unique()

# Step 2: Remove rows with those id_rand values from the business_customer_balances DataFrame
business_accounts_clean = business_accounts[~business_accounts['id_rand'].isin(business_ids_to_remove)]
business_censored_cleaned = pd.merge(business_censored, business_accounts_clean[['id_rand']], on='id_rand', how='inner')


#%% Function to segment data using Jenks algorithm and return segments
def segment_data(dataframe, n_segments=3):
    breaks = jenkspy.jenks_breaks(dataframe['time_weighted_average_balance'], n_classes=n_segments)
    segment_labels = [f'Segment {i+1}' for i in range(n_segments)]
    
    bins = pd.cut(dataframe['time_weighted_average_balance'], bins=breaks, include_lowest=True, labels=segment_labels)
    dataframe.loc[:, 'segment'] = bins
    
    segments = {label: dataframe[dataframe['segment'] == label].copy() for label in segment_labels}
    
    return segments


#%% Segment retail and business customer balances
retail_segments = segment_data(retail_accounts_clean)
business_segments = segment_data(business_accounts_clean)


#%% Merge segmentation data with original datasets
retail_full = pd.merge(
    retail_censored_cleaned, 
    retail_accounts_clean[['id_rand', 'time_weighted_average_balance', 'segment']], 
    on='id_rand', 
    how='left'
)

business_full = pd.merge(
    business_censored_cleaned, 
    business_accounts_clean[['id_rand', 'time_weighted_average_balance', 'segment']], 
    on='id_rand', 
    how='left'
)


#%% Split datasets based on segmentation
# Retail datasets
retail_segment_1 = retail_full[retail_full['segment'] == 'Segment 1']
retail_segment_2 = retail_full[retail_full['segment'] == 'Segment 2']
retail_segment_3 = retail_full[retail_full['segment'] == 'Segment 3']

# Business datasets
business_segment_1 = business_full[business_full['segment'] == 'Segment 1']
business_segment_2 = business_full[business_full['segment'] == 'Segment 2']
business_segment_3 = business_full[business_full['segment'] == 'Segment 3']


#%% Function for summary statistics
def summary_stats(df, title):
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(f"{title} Summary Statistics for 'time_weighted_average_balance':")
        print(df['time_weighted_average_balance'].describe(), "\n")
        
        unique_customer_ids = df['id_rand'].nunique()
        print(f"{title} Number of Unique Customer IDs:")
        print(unique_customer_ids, "\n")
        
        avg_transactions_per_customer = df.shape[0] / unique_customer_ids
        print(f"{title} Average Number of Transactions per Customer ID:")
        print(f"{avg_transactions_per_customer:.2f}", "\n\n")


#%% Summary statistics for the retail segments
summary_stats(retail_segment_1, "Retail Segment 1")
summary_stats(retail_segment_2, "Retail Segment 2")
summary_stats(retail_segment_3, "Retail Segment 3")


#%% Summary statistics for the business segments
summary_stats(business_segment_1, "Business Segment 1")
summary_stats(business_segment_2, "Business Segment 2")
summary_stats(business_segment_3, "Business Segment 3")


#%% Safe to parquet files
base_dir = ''

# Retail segments
retail_segment_1.to_parquet(f"{base_dir}retail_segment_1.parquet")
retail_segment_2.to_parquet(f"{base_dir}retail_segment_2.parquet")
retail_segment_3.to_parquet(f"{base_dir}retail_segment_3.parquet")

# Business segments
business_segment_1.to_parquet(f"{base_dir}business_segment_1.parquet")
business_segment_2.to_parquet(f"{base_dir}business_segment_2.parquet")
business_segment_3.to_parquet(f"{base_dir}business_segment_3.parquet")

