import pandas as pd
import warnings
import time
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import seaborn as sns

import ffn
import bt
from reporting.tools.style import set_clbrm_style

from plotly.subplots import make_subplots

def rgb_to_hex(rgb_tuple):
    return f"#{int(rgb_tuple[0] * 255):02x}{int(rgb_tuple[1] * 255):02x}{int(rgb_tuple[2] * 255):02x}"

def generate_multiplier(n):
    # Convert the number to a string and find the position of the decimal point
    n_str = str(n)
    decimal_pos = n_str.find('.')

    # If there's no decimal, the number is an integer, return 5
    if decimal_pos == -1:
        return 5

    # Find the highest non-zero decimal place
    d = 0
    for digit in n_str[decimal_pos+1:]:
        if digit != '0':
            break
        d += 1

    # Create a number with a "1" at the highest non-zero decimal place
    new_number = 1 / (10 ** (d + 1))

    # Multiply by 1
    new_number *= 5

    return new_number

default_colors = [
    (64 / 255, 75 / 255, 151 / 255),
    (154 / 255, 183 / 255, 235 / 255),
    (144 / 255, 143 / 255, 74 / 255),
    (216 / 255, 169 / 255, 23 / 255),
    (160 / 255, 84 / 255, 66 / 255),
    (189 / 255, 181 / 255, 19 / 255),
    (144 / 255, 121 / 255, 65 / 255)
]
hex_colors = [rgb_to_hex(color) for color in default_colors]
# Your original color mapping
color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Alternatives": "rgb(160, 84, 66)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)"
}

# Convert RGB strings to hexadecimal
def rgb_to_hex(rgb_str):
    rgb = rgb_str.replace('rgb', '').replace('(', '').replace(')', '').split(',')
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

hex_colors = {k: rgb_to_hex(v) for k, v in color_mapping.items()}

def percentage_formatter(x, pos):
    """Format y-axis values as percentages."""
    return f"{100 * x:.0f}%"


# Define a function to filter each group (date)
def filter_rows(group):
    max_arithmetic_mu = group['arithmetic_mu'].max() - 0.001
    sigma_at_max_mu = group.loc[group['arithmetic_mu'].idxmax(), 'sigma']

    return group[(group['arithmetic_mu'] >= max_arithmetic_mu) | (group['sigma'] <= sigma_at_max_mu)]


set_clbrm_style(caaf_colors=True)

import warnings
warnings.simplefilter(action='default', category=RuntimeWarning)

import pandas as pd

# The timestamp you used while saving
version_number = 5
country = "US"

efficient_frontier_current_caaf = pd.read_pickle(f"./data/efficient_frontier_current_caaf_{version_number}.pkl")
efficient_frontier_two_stage = pd.read_pickle(f"./data/efficient_frontier_two_stage_{version_number}.pkl")

efficient_frontier_current_caaf = efficient_frontier_current_caaf.groupby('Date').apply(filter_rows).reset_index(drop=True)
efficient_frontier_two_stage = efficient_frontier_two_stage.groupby('Date').apply(filter_rows).reset_index(drop=True)

df = efficient_frontier_two_stage
#df = efficient_frontier_current_caaf

# Filter data for a specific date, e.g., '2023-08-31'
specific_date = '1904-01-31'
specific_date_df = df[df['Date'] == specific_date]

# Select relevant columns and handle NaN values
plot_data = specific_date_df[['Target Volatility', 'Equities', 'Gov Bonds', 'Alternatives', 'HY Credit', 'Gold']].fillna(0)
plot_data['Target Volatility'] = plot_data['Target Volatility'].round(3)

plot_data = plot_data[plot_data['Target Volatility'] <= 0.166]

# Generate the color list in the order of your DataFrame columns
column_order = ['Equities', 'Gov Bonds', 'Alternatives', 'HY Credit', 'Gold']
hex_colors = [rgb_to_hex(color_mapping[col]) for col in column_order]


# Create the stacked bar chart with the specified color mapping
ax = plot_data.plot(kind='bar', x="Target Volatility", stacked=True, figsize=(10, 6), color=hex_colors)

# Adding titles and labels
plt.title(f'Asset Allocation by Target Volatility on {specific_date}')
plt.xlabel('Standard Deviation')
plt.ylabel('Weight')

# Format y-axis as percentages with 0 decimal points
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

def fmt(s):
    try:
        n = "{:.1%}".format(float(s))
    except:
        n = ""
    return n

ax.set_xticklabels([fmt(label.get_text()) for label in ax.get_xticklabels()])
# Get current x-ticks and reduce the number of ticks
xticks = ax.get_xticks()
reduced_xticks = xticks[::2]  # Keep every second tick

# Set the reduced ticks
ax.set_xticks(reduced_xticks)

# Position the legend to the right of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot with adjustments
plt.tight_layout()  # Adjust the layout
plt.show()

# Scatter plot for 'Target Volatility' vs 'Sigma'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=specific_date_df, x='Target Volatility', y='sigma')
plt.title('Target Volatility vs Sigma')
plt.xlabel('Target Volatility')
plt.ylabel('Sigma')
plt.grid(True)
plt.show()

# Scatter plot for 'Target Volatility' vs 'arithmetic_mu'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=specific_date_df, x='Target Volatility', y='arithmetic_mu')
plt.title('Target Volatility vs Arithmetic Mean (mu)')
plt.xlabel('Target Volatility')
plt.ylabel('Arithmetic Mean (mu)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=specific_date_df, x='Target Volatility', y='enb')
plt.title('Target Volatility vs ENB')
plt.xlabel('Target Volatility')
plt.ylabel('ENB')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=specific_date_df, x='Target Volatility', y='div_ratio_sqrd')
plt.title('Target Volatility vs Div Ratio Squared')
plt.xlabel('Target Volatility')
plt.ylabel('Div Ratio Squared')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=specific_date_df, x='Target Volatility', y='adjusted_md')
plt.title('Target Volatility vs Adjusted MD')
plt.xlabel('Target Volatility')
plt.ylabel('Adjusted MD')
plt.grid(True)
plt.show()


