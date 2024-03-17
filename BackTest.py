# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% Functions
def load_data(filepath):
    return pd.read_excel(filepath, index_col=0)

def calculate_L2_score(predicted_S, realized_S, T_tilde):
    predicted_S = predicted_S[predicted_S.index <= T_tilde]
    realized_S = realized_S[realized_S.index <= T_tilde]
    
    squared_diff = (realized_S.values.squeeze() - predicted_S.values.squeeze())**2
    dx = np.diff(predicted_S.index).mean()
    integral_approx = np.trapz(squared_diff, dx=dx)
    L2_score = (1 / T_tilde) * integral_approx
    return L2_score

def plot_predicted_vs_realized(predicted_S, realized_S, L2_score):
    plt.figure(figsize=(10, 7), dpi=150)
    plt.title(f'Average integrated L2 difference: {L2_score:.4f}', fontsize=16, fontweight='bold')
    plt.plot(predicted_S.index, predicted_S, color='blue', label='Predicted relative deposit volume')
    plt.plot(realized_S.index, realized_S, linestyle=':', color='red', label='Realized relative deposit volume')
    
    predicted_median = predicted_S[predicted_S <= 0.5].index[0] if not predicted_S[predicted_S <= 0.5].empty else None
    realized_median = realized_S[realized_S <= 0.5].index[0] if not realized_S[realized_S <= 0.5].empty else None
    
    plt.scatter(predicted_median, 0.5, color='blue', zorder=5, label=f'Predicted Median: {predicted_median} days')
    plt.scatter(realized_median, 0.5, color='red', zorder=5, label=f'Realized Median: {realized_median} days')
    
    plt.xlabel('Time (days)')
    plt.ylabel('Relative deposit volume')
    plt.legend(fontsize='x-large')
    plt.show()


# %% Main execution block
if __name__ == "__main__":
    # File paths (specify your file paths here)
    predicted_filepath = ''
    realized_filepath = ''

    # Load data
    predicted_S = load_data(predicted_filepath)
    realized_S = load_data(realized_filepath)
    
    # Assumptions and calculations
    T_tilde = 1826
    predicted_S.columns = ['Predicted_S']
    realized_S.columns = ['Realized_S']

    L2_score = calculate_L2_score(predicted_S, realized_S, T_tilde)
    L2_3 = round(L2_score * 100, 3)
    # plot_predicted_vs_realized(predicted_S['Predicted_S'], realized_S['Realized_S'], L2_score)

