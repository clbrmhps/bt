import os
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import numpy as np

folder_path = r"./frontiers/config_19"
files = os.listdir(folder_path)

dataframes = []

for file in files:
    file_path = os.path.join(folder_path, file)

    date_str = file.split('_')[1]
    date = pd.to_datetime(date_str, format='%Y%m%d')
    df = pd.read_pickle(file_path)
    df['date'] = date
    dataframes.append(df)

final_df = pd.concat(dataframes, ignore_index=True)

selected_date = '2024-08-31'
filtered_df = final_df[final_df['Date'] == pd.to_datetime(selected_date)]

metrics = ['Sigma', 'Diversification Ratio Squared', 'Herfindahl Index',
           'Herfindahl Index RC', 'ENB PCA', 'ENB']

def create_radar_chart(df, metrics, num_rows=10):
    num_vars = len(metrics)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    indices = np.linspace(0, len(df) - 1, num=num_rows, dtype=int)
    df_subset = df.iloc[indices]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, row in df_subset.iterrows():
        values = row[metrics].values.flatten().tolist()
        values += values[:1]

        ax.fill(angles, values, alpha=0.1, label=f'Sigma {row["Sigma"]:.2f}')
        ax.plot(angles, values, linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.title(f'Radar Chart for {num_rows} Equally Spaced Volatility Levels', size=16)
    plt.show()

create_radar_chart(filtered_df, metrics, num_rows=10)

import seaborn as sns
import matplotlib.pyplot as plt

indices = np.linspace(0, len(filtered_df) - 1, num=10, dtype=int)
df_subset = filtered_df.iloc[indices]
metrics = ['Sigma', 'Diversification Ratio Squared', 'Herfindahl Index',
           'Herfindahl Index RC', 'ENB PCA', 'ENB']

plt.figure(figsize=(10, 6))
sns.heatmap(df_subset[metrics], annot=True, cmap='coolwarm', cbar_kws={'label': 'Value'})
plt.title('Heatmap of Diversification Measures (Raw Values)')
plt.xticks(rotation=45)
plt.show()

import seaborn as sns

# Select a subset of the dataframe (up to 10 rows)
indices = np.linspace(0, len(filtered_df) - 1, num=10, dtype=int)
df_subset = filtered_df.iloc[indices]

# Metrics to visualize
metrics = ['Sigma', 'Diversification Ratio Squared', 'Herfindahl Index',
           'Herfindahl Index RC', 'ENB PCA', 'ENB']

# Create a pair plot (scatter plot matrix)
sns.pairplot(df_subset[metrics])
plt.show()

import matplotlib.pyplot as plt

# Select a subset of the dataframe (up to 10 rows)
indices = np.linspace(0, len(filtered_df) - 1, num=10, dtype=int)
df_subset = filtered_df.iloc[indices]

# Metrics to visualize
metrics = ['Sigma', 'Diversification Ratio Squared', 'Herfindahl Index',
           'Herfindahl Index RC', 'ENB PCA', 'ENB']

# Create a bar plot for each metric
fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(8, 12))
for i, metric in enumerate(metrics):
    axes[i].bar(df_subset.index, df_subset[metric])
    axes[i].set_title(metric)
    axes[i].set_xticks(df_subset.index)
    axes[i].set_xticklabels([f'{x:.2f}' for x in df_subset['Sigma']])
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Function to filter rows where Sigma is within 0.0005 of the target value
def filter_by_closest_vol(df, target_vol=0.0625, tolerance=0.0005):
    closest_rows = []

    # Group by 'Date' and filter rows based on the Sigma threshold
    for date, group in df.groupby('Date'):
        group_filtered = group[(group['Sigma'] - target_vol).abs() <= tolerance]
        if not group_filtered.empty:
            # Find the row with Sigma closest to the target_vol
            closest_row = group_filtered.iloc[(group_filtered['Sigma'] - target_vol).abs().argmin()]
            closest_rows.append(closest_row)

    # Create a new dataframe with the closest rows
    filtered_df = pd.DataFrame(closest_rows)
    return filtered_df


# Apply the filter function to get the rows with Sigma closest to 0.0625 within a tolerance of 0.0005
filtered_df = filter_by_closest_vol(final_df, target_vol=0.0625, tolerance=0.0005)

# Metrics to plot
metrics = ['Sigma', 'Diversification Ratio Squared', 'Herfindahl Index',
           'Herfindahl Index RC', 'ENB PCA', 'ENB']


# Function to transform Herfindahl metrics
def transform_herfindahl_metrics(df, metrics):
    df_transformed = df.copy()
    for metric in metrics:
        if 'Herfindahl' in metric:
            df_transformed[metric] = 1 - df_transformed[metric]
    return df_transformed


# Apply the transformation to Herfindahl metrics
filtered_df = transform_herfindahl_metrics(filtered_df, metrics)


# Create line plots for each metric with Date on the x-axis
def create_line_plots(df, metrics):
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(10, 12))

    for i, metric in enumerate(metrics):
        # Plot the metric against the Date column for the filtered vol level
        axes[i].plot(df['Date'], df[metric], marker='o', linestyle='-', label=metric)

        # Set the title and labels for each subplot
        if 'Herfindahl' in metric:
            title = f'1 - {metric} over Time (Closest Vol Level)'
        else:
            title = f'{metric} over Time (Closest Vol Level)'
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(metric)

        # Format the x-axis labels to prevent overlap
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# Call the function to create the line plots for the filtered vol level and transformed Herfindahl metrics
create_line_plots(filtered_df, metrics)

