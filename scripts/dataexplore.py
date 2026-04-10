import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This script loads the driving log data, visualizes the steering angle distribution,
# and applies a balancing technique to reduce bias in the dataset.
def load_data(csv_path='Self Driving Car/data/driving_log.csv'):
    df = pd.read_csv(csv_path,
                     names=['center','left','right',
                            'steering','throttle','brake','speed'])
    return df

# Visualize the distribution of steering angles before and after balancing
def plot_histogram(df, title='Steering Angle Distribution', save_path=None):
    plt.figure(figsize=(10, 4))
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

# Balance the dataset by limiting the number of samples in each steering angle bin
def balance_data(df, max_samples=1000):
    num_bins = 25
    # Create bins for steering angles and count samples in each bin (like a batch)
    _, bins = np.histogram(df['steering'], num_bins)
    remove_indices = []
    # For each bin, randomly remove samples if the count exceeds max_samples
    for i in range(num_bins):
        mask = (df['steering'] >= bins[i]) & (df['steering'] < bins[i+1])
        bin_indices = df[mask].index.tolist()
        if len(bin_indices) > max_samples:
            remove_indices.extend(
                np.random.choice(bin_indices,
                                 len(bin_indices) - max_samples,
                                 replace=False)
            )
    return df.drop(remove_indices).reset_index(drop=True)

# Main execution: Load data, visualize original distribution, balance data, and visualize balanced distribution
if __name__ == '__main__':
    df = load_data()
    print(f'Total samples before balancing: {len(df)}')
    plot_histogram(df, 'Original Distribution', 'Self Driving Car/docs/histogram_original.png')
    df = balance_data(df)
    print(f'Total samples after balancing:  {len(df)}')
    plot_histogram(df, 'Balanced Distribution', 'Self Driving Car/docs/histogram_balanced.png')