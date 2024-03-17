# %% Import necessary libraries
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import time


#%% Functions
def load_and_preprocess_data(filepath):
    df = pd.read_parquet(filepath)
    df['Lifetime'] = df['Lifetime'] / (60 * 60 * 24)
    columns_to_exclude = ['Start-date', 'End-date', 'Chunk ID']
    return df.drop(columns=columns_to_exclude)

def fit_cox_model(df):
    cph = CoxPHFitter()
    cph.fit(df, duration_col='Lifetime', event_col='External Withdrawal',
            cluster_col='Customer ID', strata=['Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
                                                'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12'], robust=True)
    cph.print_summary()
    cph.check_assumptions(df)
    return cph

def locate_censored_subjects(df):
    censored_subjects = df.loc[~df['External Withdrawal'].astype(bool)]
    censored_subjects_last_obs = censored_subjects['Lifetime'].astype(int)
    return censored_subjects, censored_subjects_last_obs

def predict_conditional_survival_function(cph, censored_subjects, censored_subjects_last_obs):
    times = np.arange(0, 1827)
    return cph.predict_survival_function(censored_subjects, times=times, conditional_after=censored_subjects_last_obs)

def predict_volume_runoff(censored_subjects, predictions):
    predicted_volume = pd.DataFrame({
        name: predictions[name] * row['Deposit Amount'] for name, row in censored_subjects.iterrows()
    })
    total_predicted_volume = predicted_volume.sum(axis=1)
    total_value = total_predicted_volume.max()
    return total_predicted_volume / total_value


# %% Main execution block
if __name__ == '__main__':
    filepath = ''

    # Start total execution time measurement
    total_start_time = time.time()

    # Load and preprocess data
    start_time = time.time()
    df = load_and_preprocess_data(filepath)
    print(f"Data loading and preprocessing time: {time.time() - start_time:.2f} seconds")

    # Fit Cox model
    start_time = time.time()
    cph = fit_cox_model(df)
    print(f"Cox model fitting time: {time.time() - start_time:.2f} seconds")
    
    # Locate censored objects
    start_time = time.time()
    censored_subjects, censored_subjects_last_obs = locate_censored_subjects(df)
    print(f"Locating censored subjects time: {time.time() - start_time:.2f} seconds")

    # Predict survival function
    start_time = time.time()
    predicted_conditional_survival_function = predict_conditional_survival_function(cph, censored_subjects, censored_subjects_last_obs)
    print(f"Survival function prediction time: {time.time() - start_time:.2f} seconds")

    # Calculate predicted volume
    start_time = time.time()
    predicted_volume_runoff = predict_volume_runoff(censored_subjects, predicted_conditional_survival_function)
    print(f"Predicted volume calculation time: {time.time() - start_time:.2f} seconds")

    # Save model
    predicted_volume_runoff.to_excel('')

    # Print total execution time
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")

