import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# dataexplore.py
# ---------------------------------------------------------------------------
# This module handles three responsibilities before any training occurs:
#
#   1. Loading the driving log CSV produced by the Udacity simulator.
#   2. Visualising the raw distribution of steering angles as a histogram
#      so we can inspect how balanced the collected data is.
#   3. Balancing the dataset by capping over-represented steering bins,
#      preventing the model from being biased towards driving straight
#      (steering angle ~0) which is by far the most common value when
#      driving along a relatively straight track.
#
# Why does balance matter?
#   If 80% of samples have steering ~0, the model learns to predict "drive
#   straight" for almost every input and still achieves a low training loss
#   — but fails on any curve. Capping each bin at max_samples forces the
#   model to learn from the full range of steering values.
# ---------------------------------------------------------------------------


def load_data(csv_path='Self Driving Car/data/driving_log.csv'):
    """
    Load the simulator's driving log into a Pandas DataFrame.

    The CSV has no header row, so we supply column names manually.
    Columns:
        center   - path to the centre camera image
        left     - path to the left camera image
        right    - path to the right camera image
        steering - steering angle in [-1, 1]  (negative=left, positive=right)
        throttle - throttle value in [0, 1]
        brake    - brake value in [0, 1]
        speed    - vehicle speed in mph

    For this project only 'center' (image path) and 'steering' are used.
    The left/right cameras and throttle/brake/speed columns are loaded but
    not used in training.

    Args:
        csv_path : path to driving_log.csv (default matches project structure)

    Returns:
        df : pandas DataFrame with all 7 columns
    """
    df = pd.read_csv(csv_path,
                     names=['center', 'left', 'right',
                            'steering', 'throttle', 'brake', 'speed'])
    return df


def plot_histogram(df, title='Steering Angle Distribution', save_path=None):
    """
    Plot a histogram of steering angles to visualise the data distribution.

    A red dashed horizontal line is drawn at y=1000 (the max_samples
    threshold used by balance_data) so we can visually see which bins would
    be trimmed during balancing.

    Args:
        df        : DataFrame containing a 'steering' column
        title     : title shown above the plot
        save_path : if provided, the plot is also saved to this file path
                    (useful for documenting results in the project report)
    """
    plt.figure(figsize=(10, 4))
    plt.hist(df['steering'], bins=25, color='blue')

    # Reference line at the balancing threshold — bins above this will be trimmed
    plt.axhline(y=1000, color='red', linestyle='--', label='Max threshold (1000)')

    plt.title(title)
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def balance_data(df, max_samples=1000, display=True):
    """
    Reduce over-represented steering angle bins so that no single bin
    contains more than max_samples entries.

    Algorithm:
        1. Divide the steering range into num_bins equal-width histogram bins.
        2. For each bin, identify rows whose steering value falls in that range.
        3. If a bin has more rows than max_samples, randomly select the excess
           rows and mark them for removal.
        4. Drop all marked rows and reset the DataFrame index.

    Why random removal rather than keeping the first N?
        Random removal avoids accidentally keeping only data from a specific
        time window of the recording, which could introduce temporal bias
        (e.g. always keeping laps driven in one direction).

    Args:
        df          : raw DataFrame from load_data()
        max_samples : maximum allowed samples per steering bin (default 1000)
        display     : if True, plot the balanced histogram after trimming

    Returns:
        df_balanced : new DataFrame with at most max_samples rows per bin
    """
    num_bins = 25  # Matches the 25 bins used in plot_histogram for consistency

    # np.histogram returns (counts, bin_edges); we only need the edges here
    _, bins = np.histogram(df['steering'], num_bins)

    remove_indices = []  # Accumulates row indices to drop

    for i in range(num_bins):
        # Boolean mask for rows whose steering angle falls in bin i
        # Using >= lower and < upper avoids double-counting at boundaries
        mask = (df['steering'] >= bins[i]) & (df['steering'] < bins[i + 1])
        bin_indices = df[mask].index.tolist()

        # Only trim bins that exceed the threshold
        if len(bin_indices) > max_samples:
            # Randomly pick the excess rows (those to be discarded)
            excess_count = len(bin_indices) - max_samples
            remove_indices.extend(
                np.random.choice(bin_indices, excess_count, replace=False)
            )

    # Drop all marked indices and reset index so it is contiguous again
    df_balanced = df.drop(remove_indices).reset_index(drop=True)

    if display:
        plot_histogram(df_balanced, title='Balanced Steering Angle Distribution')

    return df_balanced


# ---------------------------------------------------------------------------
# Main script: run this file directly to inspect and balance the raw dataset
# before training. The histograms are saved to the docs/ directory for
# inclusion in the project report.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    df = load_data()
    print(f'Total samples before balancing: {len(df)}')

    # Visualise raw distribution — expect a very tall spike near steering = 0
    plot_histogram(df, 'Original Distribution',
                   'Self Driving Car/docs/histogram_original.png')

    df = balance_data(df)
    print(f'Total samples after balancing:  {len(df)}')

    # Visualise balanced distribution — spike at 0 should now be capped at 1000
    plot_histogram(df, 'Balanced Distribution',
                   'Self Driving Car/docs/histogram_balanced.png')
