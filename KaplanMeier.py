#%% Import packages
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time


#%% Functions
def load_preprocess_data(filepath):
    df = pd.read_parquet(filepath)
    df['Lifetime'] = df['Lifetime'] / (60 * 60 * 24)
    df['Event'] = df['External Withdrawal'] != 0
    return df

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

def locate_censored_subjects(df):
    censored_subjects = df.loc[~df['External Withdrawal'].astype(bool)]
    censored_subjects_last_obs = censored_subjects['Lifetime'].astype(int)
    return censored_subjects, censored_subjects_last_obs

def conditional_volume_runoff_subject(args):
    index, tau_obs, max_day, volume_runoff = args
    conditional_survivals = []
    max_day = 1826

    for delta_t in range(0, max_day + 1):
        future_day = tau_obs + delta_t

        if future_day > max_day:
            future_day = max_day

        survival_future = volume_runoff.get(future_day, volume_runoff.iloc[-1])
        survival_obs = volume_runoff.get(tau_obs, volume_runoff.iloc[-1])

        conditional_survivals.append(survival_future / survival_obs)

    return index, conditional_survivals

def predict_conditional_volume_runoff(survival_series, censored_subjects, censored_subjects_last_obs, num_processes=None):
    max_day = survival_series.index.max()

    args_list = [(index, tau_obs, max_day, survival_series) for index, tau_obs in censored_subjects_last_obs.items()]

    with Pool(processes=num_processes) as pool:
        results = pool.map(conditional_volume_runoff_subject, args_list)

    conditional_survival_data = {result[0]: result[1] for result in results}
    conditional_survival_df = pd.DataFrame(conditional_survival_data, index=range(0, 1827))

    return conditional_survival_df

def predict_volume_runoff(censored_subjects, predictions):
    predicted_volume = pd.DataFrame({
        name: predictions[name] * row['Deposit Amount'] for name, row in censored_subjects.iterrows()
    })
    total_predicted_volume = predicted_volume.sum(axis=1)
    total_value = total_predicted_volume.max()
    return total_predicted_volume / total_value

def plot_predicted_volume_runoff(volume_runoff):
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
    start_time = time.time()
    df = load_preprocess_data(filepath)
    print(f"Loading and preprocessing data took: {time.time() - start_time:.2f} seconds")
    
    # Calculate Kaplan-Meier estimator and survival series
    start_time = time.time()
    kaplan_meier_df = kaplan_meier_estimator(df)
    volume_runoff = volume_runoff(kaplan_meier_df)
    print(f"Calculating Kaplan-Meier estimator and survival series took: {time.time() - start_time:.2f} seconds")

    # Locate censored objects
    start_time = time.time()
    censored_subjects, censored_subjects_last_obs = locate_censored_subjects(df)
    print(f"Locating censored subjects time: {time.time() - start_time:.2f} seconds")

    # Predict survival function
    start_time = time.time()
    predicted_conditional_volume_runoff = predict_conditional_volume_runoff(volume_runoff, censored_subjects, censored_subjects_last_obs, num_processes=7)
    print(f"Survival function prediction time: {time.time() - start_time:.2f} seconds")

    # Calculate predicted volume runoff
    start_time = time.time()
    predicted_volume_runoff = predict_volume_runoff(censored_subjects, predicted_conditional_volume_runoff)
    print(f"Predicted volume calculation time: {time.time() - start_time:.2f} seconds")
    
    # Plotting predicted volume runoff
    plot_start_time = time.time()
    median_time = plot_predicted_volume_runoff(predicted_volume_runoff)
    print(f"Plot generated in {time.time() - plot_start_time:.2f} seconds.")

    # Save model
    predicted_volume_runoff.to_excel('')

    # Print total execution time
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")