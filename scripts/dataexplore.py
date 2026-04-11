import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This script loads the driving log data, visualizes the steering angle distribution,
# and applies a balancing technique to reduce bias in the dataset.
def load_data(csv_path='./data/driving_log.csv'):
    df = pd.read_csv(csv_path,
                     names=['center','left','right',
                            'steering','throttle','brake','speed'])
    return df

# Visualize the distribution of steering angles before and after balancing
def plot_histogram(df, title='Steering Angle Distribution', save_path=None):
    plt.figure(figsize=(6, 4))
    plt.hist(df['steering'], bins=25, color='blue')
    plt.axhline(y=1000, color='red', linestyle='--', label='Max threshold')
    plt.title(title)
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Balance the dataset by limiting the number of samples in each steering angle bin, and visualize the distribution after balancing
def balance_data(df, max_samples=400, display=True):
    num_bins = 25
    # Create bins for steering angles and count samples in each bin (like a batch)
    _, bins = np.histogram(df['steering'], num_bins)
    remove_indices = [] # List to store indices of samples to remove for balancing
    # For each bin, randomly remove samples if the count exceeds max_samples
    for i in range(num_bins):
        # Create a boolean mask for the current bin, if steering angle is between bins[i] and bins[i+1], mask is True
        mask = (df['steering'] >= bins[i]) & (df['steering'] < bins[i+1])
        bin_indices = df[mask].index.tolist() # Get the indices of samples in the current bin
        # If the number of samples in the bin exceeds max_samples, randomly select indices to remove
        if len(bin_indices) > max_samples:
            remove_indices.extend(
                np.random.choice(bin_indices,
                                 len(bin_indices) - max_samples,
                                 replace=False)
            )
    
    # Remove the selected indices from the DataFrame to create a balanced dataset
    df_balanced = df.drop(remove_indices).reset_index(drop=True)
    
    # Optionally display the histogram of the balanced dataset to verify the distribution
    if display:
        plot_histogram(df_balanced, title='Balanced Steering Angle Distribution')
    return df_balanced

# Main execution: Load data, visualize original distribution, balance data, and visualize balanced distribution
if __name__ == '__main__':
    df = load_data()
    print(f'Total samples before balancing: {len(df)}')
    plot_histogram(df, 'Original Distribution', './docs/histogram_original.png')
    df = balance_data(df)
    print(f'Total samples after balancing:  {len(df)}')