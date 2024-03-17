#%% Import packages
import pandas as pd
import matplotlib.pyplot as plt
import time


#%% Functions
def load_preprocess_data(filepath):
    df = pd.read_parquet(filepath)
    mask = df['Start-date'] == pd.Timestamp('2019-01-01')
    df_censored = df[mask].copy()
    df_censored['Lifetime'] = df_censored['Lifetime'] / (60 * 60 * 24)
    df_censored['Event'] = df_censored['External Withdrawal'] != 0
    return df_censored

def kaplan_meier_estimator(data):
    df_sorted = data.sort_values(by='Lifetime')
    unique_times = df_sorted.loc[df_sorted['Event'], 'Lifetime'].unique()
    S_t = 1
    kaplan_meier_results = []

    for time_point in unique_times:
        n_j = df_sorted[df_sorted['Lifetime'] >= time_point]['Deposit Amount'].sum()
        d_j = df_sorted[(df_sorted['Lifetime'] == time_point) & (df_sorted['Event'])]['Deposit Amount'].sum()

        if n_j > 0:
            S_t *= (1 - d_j / n_j)
        kaplan_meier_results.append((time_point, S_t))

    return pd.DataFrame(kaplan_meier_results, columns=['Time', 'S(t)'])

def volume_runoff(kaplan_meier_df):
    max_day = 1826
    volume_runoff = pd.Series(index=range(0, max_day + 1), dtype=float)

    if not kaplan_meier_df.empty:
        current_survival_prob = kaplan_meier_df.iloc[0]['S(t)']
    else:
        current_survival_prob = 1.0

    for day in volume_runoff.index:
        applicable_times = kaplan_meier_df[kaplan_meier_df['Time'] <= day]
        if not applicable_times.empty:
            current_survival_prob = applicable_times.iloc[-1]['S(t)']
        volume_runoff[day] = current_survival_prob
    
    return volume_runoff

def plot_volume_runoff(volume_runoff):
    plt.figure(figsize=(10, 7), dpi=150)
    volume_runoff.plot()
    plt.xlabel('Time (days)')
    plt.ylabel('Relative deposit volume')

    median_time = volume_runoff[volume_runoff <= 0.5].index[0] if not volume_runoff[volume_runoff <= 0.5].empty else None
    if median_time is not None:
        plt.axvline(x=median_time, color='r', linestyle='--', label=f'Median: {median_time} days')

    specific_days = [30, 1826]
    colors = ['blue', 'green']
    labels = ['1 Month', '5 Years']
    for day, color, label in zip(specific_days, colors, labels):
        if day in volume_runoff.index:
            volume_value = volume_runoff[day]
            plt.scatter(day, volume_value, color=color, zorder=5, label=f'Relative deposit volume after {label}: {volume_value:.2f}%')

    plt.legend(fontsize='x-large')
    plt.show()
    return median_time


#%% Main execution
if __name__ == "__main__":
    filepath = ''
    
    # Start total execution time measurement
    total_start_time = time.time()
    
    # Load and preprocess data
    preprocess_start_time = time.time()
    df = load_preprocess_data(filepath)
    print(f"Data preprocessed in {time.time() - preprocess_start_time:.2f} seconds.")

    # Kaplan-Meier Estimator
    km_start_time = time.time()
    kaplan_meier_df = kaplan_meier_estimator(df)
    print(f"Kaplan-Meier estimator computed in {time.time() - km_start_time:.2f} seconds.")

    # Volume runoff
    volume_runoff_start_time = time.time()
    volume_runoff = volume_runoff(kaplan_meier_df)
    print(f"Survival series generated in {time.time() - volume_runoff_start_time:.2f} seconds.")
    
    # Plotting Volume Runoff
    plot_start_time = time.time()
    median_time = plot_volume_runoff(volume_runoff)
    print(f"Plot generated in {time.time() - plot_start_time:.2f} seconds.")
    
    # Save model
    volume_runoff.to_excel('/realized_R1_long.xlsx')
    
    # Total Execution Time
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds.")

