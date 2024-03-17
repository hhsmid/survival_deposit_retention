#%% Import required packages
import numpy as np
import pandas as pd
from pandas import Timestamp
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial


#%% Functions
# Functions for data loading
def load_data(transaction_data_path, external_data_path):
    transaction_df = pd.read_parquet(transaction_data_path)
    external_df = pd.read_excel(external_data_path)
    return transaction_df, external_df

def merge_same_second_transactions(df):
    net_transactions = df.groupby(['id_rand', 'transaction_timestamp'])['amount'].sum().reset_index()
    net_transactions = net_transactions[net_transactions['amount'] != 0]
    
    balance_before = df.groupby(['id_rand', 'transaction_timestamp']).apply(lambda x: x.iloc[0]['balance_before']).reset_index(name='balance_before')
    balance_after = df.groupby(['id_rand', 'transaction_timestamp']).apply(lambda x: x.iloc[-1]['balance_after']).reset_index(name='balance_after')
    
    merged = pd.merge(net_transactions, balance_before, on=['id_rand', 'transaction_timestamp'], how='left')
    merged = pd.merge(merged, balance_after, on=['id_rand', 'transaction_timestamp'], how='left')

    return merged

def merge_transactions_parallel(df):
    result = merge_same_second_transactions(df)
    return result

def clean_and_prepare_data(df):
    sampled_ids = df['id_rand'].drop_duplicates().sample(n=50, random_state=1)
    df = df[df['id_rand'].isin(sampled_ids)]

    df_sorted = df.sort_values(by=['id_rand', 'transaction_timestamp'])

    partitions = [group for _, group in df_sorted.groupby('id_rand')]

    with ProcessPoolExecutor(max_workers=7) as executor:
        results = list(executor.map(merge_transactions_parallel, partitions))

    combined_df = pd.concat(results, ignore_index=True)
    return combined_df, sampled_ids


# Functions for deposit chunk creation
def create_balance_layers(df, id_rand):
    df_filtered = df[df['id_rand'] == id_rand]
    unique_balance_afters = np.unique(df_filtered['balance_after'].values)
    first_balance_before = df_filtered['balance_before'].iloc[0]
    balances = np.unique(np.concatenate(([first_balance_before], unique_balance_afters)))
    sorted_balances = np.sort(balances)
    layers = {}
    start_balance = 0.0
    layer_id = 1
    for balance in sorted_balances:
        if balance > start_balance:
            layers[layer_id] = (start_balance, balance)
            start_balance = balance
            layer_id += 1
    return layers

def create_chunks(df, id_rand, start_date, censoring_date):
    layers = create_balance_layers(df, id_rand)
    df_sorted = df[df['id_rand'] == id_rand].sort_values(by='transaction_timestamp')
    
    initial_balance = df_sorted.iloc[0]['balance_before']
    
    start_date = Timestamp(start_date)
    censoring_date = Timestamp(censoring_date)

    next_section_id = {layer_id: 1 for layer_id in layers.keys()}
    
    active_sections = {}
    for layer_id, (lower_bound, upper_bound) in layers.items():
        if initial_balance >= upper_bound:
            active_sections[layer_id] = {
                'section_id': next_section_id[layer_id],
                'start_date': start_date,
                'end_date': None
            }
            next_section_id[layer_id] += 1
    
    chunk_data = []

    for index, row in df_sorted.iterrows():
        transaction_date = Timestamp(row['transaction_timestamp'])
        balance_after = row['balance_after']
        transaction_amount = row['amount']

        for layer_id, (lower_bound, upper_bound) in layers.items():
            if layer_id in active_sections and transaction_amount < 0 and balance_after <= lower_bound:
                section_info = active_sections[layer_id]
                lifetime = (transaction_date - section_info['start_date']).total_seconds()
                chunk_data.append({
                    'Customer ID': id_rand,
                    'Chunk ID': f"({layer_id}, {section_info['section_id']})",
                    'Deposit Amount': upper_bound - lower_bound,
                    'Lifetime': lifetime,
                    'External Withdrawal': 1,
                    'Start-date': section_info['start_date'],
                    'End-date': transaction_date
                })
                del active_sections[layer_id]

            elif transaction_amount > 0 and balance_after >= upper_bound:
                if layer_id not in active_sections:
                    active_sections[layer_id] = {
                        'section_id': next_section_id[layer_id],
                        'start_date': transaction_date,
                        'end_date': None
                    }
                    next_section_id[layer_id] += 1

    for layer_id, (lower_bound, upper_bound) in layers.items():
        if layer_id in active_sections:
            section_info = active_sections[layer_id]
            lifetime = (censoring_date - section_info['start_date']).total_seconds()
            chunk_data.append({
                'Customer ID': id_rand,
                'Chunk ID': f"({layer_id}, {section_info['section_id']})",
                'Deposit Amount': upper_bound - lower_bound,
                'Lifetime': lifetime,
                'External Withdrawal': 0,
                'Start-date': section_info['start_date'],
                'End-date': censoring_date
            })
    
    return pd.DataFrame(chunk_data)


# Functions for data aggregation
def process_chunk(chunk_ids, transaction_df, start_date, censoring_date):
    return pd.concat([create_chunks(transaction_df, id_rand, start_date, censoring_date) for id_rand in chunk_ids], ignore_index=True)

def parallel_aggregate_chunks(transaction_df, start_date, censoring_date, chunk_size=50):
    unique_ids = transaction_df['id_rand'].unique()

    chunks = [unique_ids[i:i + chunk_size] for i in range(0, len(unique_ids), chunk_size)]

    process_chunk_with_data = partial(process_chunk, transaction_df=transaction_df, start_date=start_date, censoring_date=censoring_date)
    
    with ProcessPoolExecutor(max_workers=7) as executor:
        aggregated_dfs = list(executor.map(process_chunk_with_data, chunks))
    return pd.concat(aggregated_dfs, ignore_index=True)


# Functions for merging survival data with external time series data
def expand_transactions_chunk(combined_df, external_df):
    external_df['Date'] = pd.to_datetime(external_df['Date']).dt.to_period('M').dt.to_timestamp()
    expanded_rows = []
    last_chunk_id = None
    segment_counter = 0
    t_start = 0

    for _, row in combined_df.iterrows():
        start_date = pd.to_datetime(row['Start-date'])
        end_date = pd.to_datetime(row['End-date'])
        current_date = start_date
        current_chunk_id = row['Chunk ID']

        if current_chunk_id != last_chunk_id:
            t_start = 0 
            segment_counter = 1
        else:
            pass

        while current_date <= end_date:
            end_date_split = (pd.Timestamp(current_date.year, current_date.month, 1) + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
            if current_date.month == end_date.month and current_date.year == end_date.year:
                end_date_split = end_date

            lifetime_split = max(0, (end_date_split - current_date).total_seconds() / (24 * 3600))
            
            t_stop = t_start + lifetime_split

            month_year = current_date.to_period('M').to_timestamp()

            expanded_row = {
                'Customer ID': row['Customer ID'],
                'Chunk ID': row['Chunk ID'],
                'Segmentation Counter': segment_counter,
                'Deposit Amount': row['Deposit Amount'],
                'Lifetime': lifetime_split,
                'External Withdrawal': row['External Withdrawal'],
                'MonthYear': month_year,
                't-start': t_start,
                't-stop': t_stop,
                'event': 0
            }
            
            if end_date_split == end_date and row['External Withdrawal'] == 1:
                expanded_row['event'] = 1 

            expanded_rows.append(expanded_row)

            t_start = t_stop
            current_date = end_date_split + pd.Timedelta(days=1)
            segment_counter = segment_counter + 1

        last_chunk_id = current_chunk_id

    expanded_df = pd.DataFrame(expanded_rows)
    merged_df = pd.merge(expanded_df, external_df, left_on='MonthYear', right_on='Date', how='left')
    merged_df.drop(['MonthYear', 'Date'], axis=1, inplace=True)

    return merged_df

def parallel_merge_chunks_covariates_counting(combined_df, external_df):
    """Expands and merges data with external covariate monthly data in parallel."""
    unique_customer_ids = combined_df['Customer ID'].unique()
    chunks = [combined_df[combined_df['Customer ID'] == cid] for cid in unique_customer_ids]
    
    expanded_chunks = []
    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(expand_transactions_chunk, chunk, external_df) for chunk in chunks]
        for future in futures:
            expanded_chunks.append(future.result())
    
    return pd.concat(expanded_chunks, ignore_index=True)


#%% Execution block
if __name__ == '__main__':
    # File paths
    fp_transaction_data = ''
    fp_external_data = '/covariates.xlsx'

    # Load data
    start_time = time.time()
    transaction_df, external_df = load_data(fp_transaction_data, fp_external_data)
    print(f"Loading data took: {time.time() - start_time} seconds")

    # Clean and prepare data
    start_time = time.time()
    transaction_df, sampled_ids = clean_and_prepare_data(transaction_df)
    print(f"Cleaning data took: {time.time() - start_time} seconds")

    # Create chunk data and aggregate over id's in parallel
    start_time = time.time()
    aggregated_chunks_df = parallel_aggregate_chunks(transaction_df, '2014-01-01', '2019-01-01')
    print(f"Aggregating data took: {time.time() - start_time} seconds")
    
    # Find censored sampled ids
    start_time = time.time()
    censored_sampled_ids = aggregated_chunks_df[aggregated_chunks_df['External Withdrawal'] == 0]['Customer ID'].unique()
    censored_sampled_ids_df = pd.DataFrame(censored_sampled_ids, columns=['id_rand'])
    print(f"Finding censored sampled ID's took: {time.time() - start_time} seconds")
    
    # Expand and merge data in parallel
    start_time = time.time()
    chunks_covariates_merged_df = parallel_merge_chunks_covariates_counting(aggregated_chunks_df, external_df)
    print(f"Merging data took: {time.time() - start_time} seconds")

    # Save the final DataFrame
    chunks_covariates_merged_df.to_parquet('')
    censored_sampled_ids_df.to_excel('', index=False)
