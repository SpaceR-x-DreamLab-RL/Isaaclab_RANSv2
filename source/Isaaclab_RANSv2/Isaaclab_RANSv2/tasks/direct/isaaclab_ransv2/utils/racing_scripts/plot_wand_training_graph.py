import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mean_and_std_from_csvs(file_paths, smoothing_window=50):
    """
    Reads multiple CSV files, identifies 'mean_reward' columns (excluding MIN/MAX),
    calculates the mean and standard deviation across these columns for each step,
    applies a rolling mean for smoothing, and plots them on a single graph.

    Args:
        file_paths (list): A list of strings, where each string is the path
                           to a CSV file containing the robot's performance data.
                           Each CSV file is expected to have a 'Step' column
                           and multiple 'mean_reward' columns (e.g., from different seeds).
        smoothing_window (int): The window size for the rolling mean to smooth the
                                mean reward curve. A larger window results in more smoothing.
    """
    # Set up the plot aesthetics
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid') # A visually appealing style

    # Iterate through each provided CSV file
    for file_path in file_paths:
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Identify columns that represent 'mean_reward' for different seeds/runs.
            # We look for columns containing '/mean_reward' but explicitly exclude
            # those ending with '__MIN' or '__MAX' as these are typically bounds,
            # not raw run data for calculating overall mean/std.
            mean_reward_cols = [
                col for col in df.columns
                if '/mean_reward' in col and '__MIN' not in col and '__MAX' not in col
            ]

            # Check if any relevant columns were found
            if not mean_reward_cols:
                print(f"Warning: No 'mean_reward' columns found (excluding MIN/MAX) in '{file_path}'. Skipping this file.")
                continue

            # Ensure the 'Step' column exists for the x-axis
            if 'Step' not in df.columns:
                print(f"Error: 'Step' column not found in '{file_path}'. Skipping this file.")
                continue

            # Convert identified mean_reward columns to numeric, coercing any errors
            # (e.g., empty strings or non-numeric values) to NaN.
            for col in mean_reward_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where 'Step' is NaN or where all identified mean_reward columns are NaN.
            # This ensures we only process valid data points.
            df_cleaned = df.dropna(subset=['Step'] + mean_reward_cols, how='all')

            if df_cleaned.empty:
                print(f"Warning: After cleaning, no valid data rows remain in '{file_path}'. Skipping this file.")
                continue

            # Filter the DataFrame to include only the first 1000 steps
            df_filtered = df_cleaned[df_cleaned['Step'] <= 1000]

            if df_filtered.empty:
                print(f"Warning: After filtering for steps <= 1000, no data remains in '{file_path}'. Skipping this file.")
                continue

            # Calculate the row-wise mean of the 'mean_reward' columns.
            # This gives us the average reward across all seeds for each 'Step'.
            mean_rewards = df_filtered[mean_reward_cols].mean(axis=1)

            # Calculate the row-wise standard deviation of the 'mean_reward' columns.
            # This quantifies the variability of rewards across seeds for each 'Step'.
            std_rewards = df_filtered[mean_reward_cols].std(axis=1)

            # Apply rolling mean to smooth the mean_rewards curve
            # min_periods=1 ensures that smoothing starts from the first data point
            smoothed_mean_rewards = mean_rewards.rolling(window=smoothing_window, min_periods=1).mean()

            # Get the 'Step' values for the x-axis
            steps = df_filtered['Step']

            # Create a label for the plot based on the filename
            # This helps differentiate plots if you have multiple CSVs
            plot_label = os.path.basename(file_path).replace('.csv', '')

            # Plot the smoothed mean reward line
            plt.plot(steps, smoothed_mean_rewards, label=f'Smoothed Mean Reward (Window={smoothing_window}): {plot_label}', linewidth=2)

            # Plot the shaded area representing one standard deviation above and below the mean
            # Note: The standard deviation is not smoothed here, it reflects the raw variability.
            plt.fill_between(steps, smoothed_mean_rewards - std_rewards, smoothed_mean_rewards + std_rewards,
                             alpha=0.2, label=f'Std Dev: {plot_label}')

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Please check the path.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{file_path}' is empty. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file_path}': {e}")

    # Add labels, title, and legend to the plot for clarity
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.title(f'Robot Rendezvous Performance: Smoothed Mean Reward over Training Steps (First 1000 Steps, Window={smoothing_window})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
    plt.show()


csv_files_to_plot = [
    'wandb_export_2025-07-24T16_44_33.120+02_00.csv', 
    'wandb_export_2025-07-24T16_51_29.483+02_00.csv', 
    'wandb_export_2025-07-25T09_05_07.751+02_00.csv',
    'wandb_export_2025-07-25T09_07_19.537+02_00.csv',
    'wandb_export_2025-07-25T09_08_21.516+02_00.csv',
    'wandb_export_2025-07-25T09_10_47.371+02_00.csv',
]

plot_mean_and_std_from_csvs(csv_files_to_plot)