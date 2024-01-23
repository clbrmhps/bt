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
from scipy.optimize import minimize

import seaborn as sns
from ffn.core import calc_erc_weights

import ffn
import bt
from reporting.tools.style import set_clbrm_style

from plotly.subplots import make_subplots

from meucci.EffectiveBets import EffectiveBets
from meucci.torsion import torsion

from portfolio_construction.calculations import pf_mu
from portfolio_construction.calculations import pf_sigma

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

def diversification_ratio_squared(w, sigma_port, standard_deviations):
    return (np.dot(w, standard_deviations) / sigma_port) ** 2

def weight_objective(weight, weight_ref, norm, delta=0.5):
        weight = weight.reshape(-1, 1)
        weight_ref = weight_ref.reshape(-1, 1)

        diff = weight - weight_ref
        if norm == 'l1':
            return np.sum(np.abs(diff))
        elif norm == 'l2':
            return np.dot(diff.T, diff).squeeze()
        elif norm == 'huber':
            abs_diff = np.abs(diff)
            is_small_error = abs_diff <= delta
            squared_loss = 0.5 * (diff ** 2)
            linear_loss = delta * (abs_diff - 0.5 * delta)
            return np.sum(np.where(is_small_error, squared_loss, linear_loss))

set_clbrm_style(caaf_colors=True)

import warnings
warnings.simplefilter(action='default', category=RuntimeWarning)

import pandas as pd

# The timestamp you used while saving
version_number = 5
country = "US"

rdf = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf.set_index('Date', inplace=True)
# rdf.dropna(inplace=True)

er = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="expected_gross_return")
er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
er.set_index('Date', inplace=True)
er.loc[:"1973-01-31", "Gold"] = np.nan

const_covar = rdf.cov()
covar = const_covar * 12

selected_date = "1981-01-31"
selected_assets = ["Equities", "HY Credit", "Gov Bonds", "Gold", "Alternatives"]

selected_covar = covar.loc[selected_assets, selected_assets]
selected_er = er.loc[selected_date, selected_assets]

erc_weights = calc_erc_weights(returns=rdf,
                               initial_weights=None,
                               risk_weights=None,
                               covar_method="constant",
                               risk_parity_method="slsqp",
                               maximum_iterations=100,
                               tolerance=1e-4,
                               const_covar=covar.loc[selected_assets, selected_assets])

stds = np.sqrt(np.diag(selected_covar))
arithmetic_mu = selected_er + np.square(stds) / 2

efficient_frontier_current_caaf = pd.read_pickle(f"./data/efficient_frontier_current_caaf_{version_number}.pkl")
efficient_frontier_two_stage = pd.read_pickle(f"./data/efficient_frontier_two_stage_{version_number}.pkl")

selected_config_ts = efficient_frontier_two_stage.loc[(efficient_frontier_two_stage.loc[:, "Date"]==selected_date) & (efficient_frontier_two_stage.loc[:, "sigma"]>=0.069), :].iloc[0, :]
selected_weights_ts_pd = selected_config_ts.loc[selected_assets]
selected_weights_ts = selected_config_ts.loc[selected_assets].to_numpy().astype(np.float64)

selected_config_cc = efficient_frontier_current_caaf.loc[(efficient_frontier_current_caaf.loc[:, "Date"]==selected_date) & (efficient_frontier_current_caaf.loc[:, "sigma"]>=0.070), :].iloc[0, :]
selected_weights_cc_pd = selected_config_cc.loc[selected_assets]
selected_weights_cc = selected_config_cc.loc[selected_assets].to_numpy().astype(np.float64)

sigma_port_cc = pf_sigma(selected_weights_cc, selected_covar.to_numpy())
sigma_port_ts = pf_sigma(selected_weights_ts, selected_covar.to_numpy())

mu_port_cc = pf_mu(selected_weights_cc, arithmetic_mu)
mu_port_ts = pf_mu(selected_weights_ts, arithmetic_mu)

t_mt_cc = torsion(selected_covar, 'minimum-torsion', method='exact')
p_cc, enb_cc = EffectiveBets(selected_weights_cc, selected_covar.to_numpy(), t_mt_cc)

t_mt_ts = torsion(selected_covar, 'minimum-torsion', method='exact')
p_ts, enb_ts = EffectiveBets(selected_weights_ts, selected_covar.to_numpy(), t_mt_ts)

diversification_ratio_squared_cc = diversification_ratio_squared(selected_weights_cc, sigma_port_cc, stds)
diversification_ratio_squared_ts = diversification_ratio_squared(selected_weights_ts, sigma_port_ts, stds)

####################################################################################################
# Sampled Points

def generate_weights(minimum, maximum, increment, target):
    if (minimum.shape != maximum.shape) or (np.ndim(minimum) != 1):
        raise Exception('minimum and maximum must be of dimension 1 and of same length.')
    if np.any(minimum > maximum):
        raise Exception('minima cannot be larger than maxima.')
    if not np.all(np.isclose(np.mod(minimum / increment, 1), 0)) or \
            not np.all(np.isclose(np.mod(maximum / increment, 1), 0)):
        raise Exception('minima and maxima must be divisible by increment.')
    minimum = (minimum / increment).astype(int)
    maximum = (maximum / increment).astype(int)
    target = int(target / increment)
    # Loop
    final_list = list()
    for j in range(minimum[0], maximum[0] + 1):
        final_list.append((j,))
    for i in range(1, len(minimum)):
        temp = list()
        for current in final_list:
            for j in range(max(target - sum(current) - sum(maximum[i + 1:]), minimum[i]),
                           min(target - sum(current) - sum(minimum[i + 1:]), maximum[i]) + 1):
                temp.append(current + (j,))
        final_list = temp
    return np.array(final_list) * increment

# Make it so that er_aligned has exactly the same missingness pattern as rdf
er_aligned = er.reindex(rdf.index)
er_aligned.loc[er_aligned.index.difference(rdf.index)] = np.nan
for col in er.columns:
    er_aligned.loc[rdf[col].isna(), col] = np.nan

# Initialize variables to hold start point, list of results, and list of asset blocks
start_point = 0
block_expected_returns = []
date_indexes = []
block_stdevs = []
block_weights = []

# Iterate through the DataFrame to identify blocks and multiply
for i in range(1, len(er_aligned)):
    # Identify the dimensions of non-NA values for the current and previous rows
    curr_dims = er_aligned.iloc[i].dropna().shape[0]
    prev_dims = er_aligned.iloc[i - 1].dropna().shape[0]

    # Check if the dimensions change, indicating the start of a new block
    if curr_dims != prev_dims:
        # Extract the block and drop NA values along columns
        block = er_aligned.iloc[start_point:i].dropna(axis=1)

        # Record the assets in this block
        assets_in_block = block.columns.tolist()

        # Generate the weight matrix w for this block
        min_vals = np.zeros(block.shape[1])
        max_vals = np.ones(block.shape[1])
        w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)

        aligned_covar = const_covar.loc[assets_in_block, assets_in_block] * 12
        block_arithmetic_return = block.values + np.diag(aligned_covar.to_numpy())/2

        # Perform the matrix multiplication and append to the result list
        block_result = block_arithmetic_return.dot(w.transpose())

        for j, date_index in enumerate(block.index):
            if j == 0:
                continue
            date_indexes.append(date_index)
            block_weights.append(w)
            block_expected_returns.append(block_result[j])

        # Generate w for the block (reuse your generate_weights function)
        # min_vals = np.zeros(len(assets_in_block))
        # max_vals = np.ones(len(assets_in_block))
        # w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)

        # Calculate the standard deviation for each weight vector in w
        stdevs = []
        for w_row in w:
            stdev = np.sqrt(w_row @ aligned_covar @ w_row.T)  # @ is matrix multiplication
            stdevs.append(stdev)

        for j, date_index in enumerate(block.index):
            block_stdevs.append(stdevs)

        # Update the start_point for the next block
        start_point = i

# Handle the last block after exiting the loop
block = er_aligned.iloc[start_point:].dropna(axis=1)

# Record the assets in this block
assets_in_block = block.columns.tolist()

min_vals = np.zeros(block.shape[1])
max_vals = np.ones(block.shape[1])
w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)
w = generate_weights(minimum=min_vals, maximum=max_vals, increment=0.025, target=1)

aligned_covar = const_covar.loc[assets_in_block, assets_in_block] * 12
block_arithmetic_return = block.values + np.diag(aligned_covar.to_numpy()) / 2

block_result = block_arithmetic_return.dot(w.transpose())
for j, date_index in enumerate(block.index):
    print(date_index)
    date_indexes.append(date_index)
    block_expected_returns.append(block_result[j])
    block_weights.append(w)

# Generate w for the block (reuse your generate_weights function)
# min_vals = np.zeros(len(assets_in_block))
# max_vals = np.ones(len(assets_in_block))
# w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)

# Calculate the standard deviation for each weight vector in w
stdevs = []
for w_row in w:
    stdev = np.sqrt(w_row @ aligned_covar @ w_row.T)  # @ is matrix multiplication
    stdevs.append(stdev)

for j, date_index in enumerate(block.index):
    block_stdevs.append(stdevs)

# import matplotlib.pyplot as plt
#
# # Loop through each date
# for i, (returns_matrix, stdevs) in enumerate(zip(block_expected_returns, block_stdevs)):
#
#     # Loop through the 348 different asset sets for a single date
#     for j in range(returns_matrix.shape[0]):
#         plt.figure(figsize=(10, 6))
#
#         # Grab the jth set of expected returns for this date
#         returns = returns_matrix[j, :]
#
#         # Plot the efficient frontier for this asset set and date
#         plt.scatter(stdevs, returns, c='blue', label='Efficient Frontier')
#
#         # Optional: add labels, titles, etc.
#         plt.xlabel('Standard Deviation')
#         plt.ylabel('Expected Return')
#         plt.title(f'Efficient Frontier for Date Index {i}, Asset Set {j + 1}')
#         plt.grid(True)
#         plt.legend(loc='upper left')
#
#         # Show plot
#         plt.show()

max_stdev = max(list(chain.from_iterable(block_stdevs)))
min_stdev = min(list(chain.from_iterable(block_stdevs)))

####################################################################################################
# Weights

# Renaming for clarity
weights_df = pd.DataFrame({
    'Current CAAF': selected_weights_cc_pd,
    'Two Stage': selected_weights_ts_pd
}).T  # Transpose to switch rows and columns

# Plotting
ax = weights_df.plot(kind='bar', stacked=True, color=[hex_colors.get(x, '#333333') for x in weights_df.columns])
plt.title(f'Weights on {selected_date}')
plt.xlabel('Methodology')
plt.ylabel('Weights')

# Formatting Y-axis as percentage
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Moving the legend to the right of the chart
plt.legend(title='Asset Class', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to accommodate the legend
plt.tight_layout()

plt.xticks(rotation=0)  # Keeping the x-axis labels horizontal
plt.show()

####################################################################################################
# Diversification Metrics
def weight_objective(weight, weight_ref, norm, delta=0.5):
    weight = weight.reshape(-1, 1)
    weight_ref = weight_ref.reshape(-1, 1)

    diff = weight - weight_ref
    if norm == 'l1':
        return np.sum(np.abs(diff))
    elif norm == 'l2':
        return np.dot(diff.T, diff).squeeze()
    elif norm == 'huber':
        abs_diff = np.abs(diff)
        is_small_error = abs_diff <= delta
        squared_loss = 0.5 * (diff ** 2)
        linear_loss = delta * (abs_diff - 0.5 * delta)
        return np.sum(np.where(is_small_error, squared_loss, linear_loss))

# Create a DataFrame
data = {
    'Current CAAF': [enb_cc.item(), diversification_ratio_squared_cc],
    'Two Stage': [enb_ts.item(), diversification_ratio_squared_ts]
}
metrics_df = pd.DataFrame(data, index=['ENB', 'Diversification Ratio Squared'])
ax = metrics_df.plot(kind='bar')
plt.title('Comparison of Metrics by Methodology')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(rotation=0)  # To keep the x-axis labels horizontal for clarity
plt.show()

####################################################################################################
# Two Stage Optimization

def return_objective(weights, exp_rets):
    # portfolio mean
    mean = sum(exp_rets * weights)
    # negative because we want to maximize the portfolio mean
    # and the optimizer minimizes metric
    return mean

def volatility_constraint(weights, covar, target_volatility):
    # portfolio volatility
    port_vol = np.sqrt(np.dot(np.dot(weights, covar), weights))
    # we want to ensure our portfolio volatility is equal to the given number
    return port_vol - target_volatility

weight_ref = np.array(erc_weights[selected_assets])
n = len(selected_assets)
initial_weights_two_stage = np.ones([n]) / n

norm = "huber"
bounds = [(0, 1) for i in range(n)]
epsilon = 0.1

target_return = 0.064
target_volatility = 0.07

constraints = [
    {"type": "eq", "fun": lambda w: sum(w) - 1.0},
    {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return},
    {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_volatility)}
]

optimum = minimize(
    weight_objective, x0=initial_weights_two_stage, args=(weight_ref, norm), method='SLSQP',
    constraints=constraints, bounds=bounds,
    options={'maxiter': 1000, 'ftol': 1e-10}
)

# Convert string to Timestamp
selected_date_ts = pd.Timestamp(selected_date)

# Find the index of the selected date in the list
if selected_date_ts in date_indexes:
    selected_index = date_indexes.index(selected_date_ts)
    print(f"The index of {selected_date} is {selected_index}")
else:
    print(f"{selected_date} not found in the list")

selected_returns = block_expected_returns[selected_index]
selected_stdevs = block_stdevs[selected_index]
selected_weights = block_weights[selected_index]

# Create a DataFrame for returns and stdevs
df = pd.DataFrame({
    "Returns": selected_returns,
    "Stdevs": selected_stdevs
})

# Add each column of selected_weights to the DataFrame
for i in range(selected_weights.shape[1]):
    df[selected_assets[i]] = selected_weights[:, i]

target_stdev = 0.07
tolerance = 0.00005

close_to_target = df[abs(df['Stdevs'] - target_stdev) <= tolerance]

def calculate_enb(row, assets, covariance_matrix, t_mt):
    # Assuming EffectiveBets function takes weights and covariance matrix as inputs
    try:
        covariance_matrix = covariance_matrix.to_numpy()
    except AttributeError:
        pass

    weights = row[assets].to_numpy()
    _, enb = EffectiveBets(weights, covariance_matrix, t_mt)  # or t_mt_ts
    return enb.item()

def calculate_div_ratio_squared(row, assets, sigma_port, stds):
    # Assuming diversification_ratio_squared function takes weights, sigma_port, and stds as inputs
    weights = row[assets].to_numpy()
    div_ratio_squared = diversification_ratio_squared(weights, sigma_port, stds)  # or sigma_port_ts
    return div_ratio_squared

# Apply the functions to each row
close_to_target['ENB'] = close_to_target.apply(lambda x: calculate_enb(x, selected_assets, selected_covar, t_mt_cc), axis=1)
close_to_target['Diversification Ratio Squared'] = close_to_target.apply(lambda x: calculate_div_ratio_squared(x, selected_assets, sigma_port_cc, stds), axis=1)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Plot for ENB
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Stdevs', y='Returns', hue='ENB', data=close_to_target, palette='viridis', alpha=0.6, edgecolor='w')
plt.title('Jitter Plot with ENB')
plt.xlabel('Standard Deviations')
plt.ylabel('Returns')
plt.legend(title='ENB', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import plotly.express as px

# Assuming 'close_to_target' is your DataFrame
fig = px.scatter(
    close_to_target,
    x='Stdevs',
    y='Returns',
    color='ENB',
    hover_data=['Equities', 'Gov Bonds', 'Alternatives'],  # Columns to show in hover tooltip
    title='Jitter Plot with ENB',
    labels={'Stdevs': 'Standard Deviations', 'Returns': 'Returns', 'ENB': 'ENB'},  # Custom labels
    color_continuous_scale='Viridis'  # Color scale similar to 'viridis'
)

# Update layout
fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(title='Standard Deviations'),
    yaxis=dict(title='Returns'),
    legend_title_text='ENB',
    margin=dict(l=60, r=60, t=60, b=60)
)

# Show plot
fig.show()

fig.write_html("scatter_plot.html")


# Plot for Diversification Ratio Squared
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Stdevs', y='Returns', hue='Diversification Ratio Squared', data=close_to_target, palette='magma', alpha=0.6, edgecolor='w')
plt.title('Jitter Plot with Diversification Ratio Squared')
plt.xlabel('Standard Deviations')
plt.ylabel('Returns')
plt.legend(title='Diversification Ratio Squared', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

sorted_df = close_to_target.sort_values(by='ENB', ascending=False)
selected_weights_ts_pd = selected_weights_ts_pd.astype(np.float64)
selected_weights_cc_pd = selected_weights_cc_pd.astype(np.float64)
calculate_enb(selected_weights_ts_pd, selected_assets, selected_covar, t_mt_ts)
calculate_enb(selected_weights_cc_pd, selected_assets, selected_covar, t_mt_cc)

# Define a range of target standard deviations
target_stdevs = np.arange(0.05, 0.10, 0.0025)  # Adjust this range as needed
tolerance = 0.00005

# Initialize a DataFrame to store the results
results_df = pd.DataFrame(columns=['Target Stdev', 'Max ENB'])

for target_stdev in target_stdevs:
    close_to_target = df[abs(df['Stdevs'] - target_stdev) <= tolerance]
    target_return = close_to_target['Returns'].max()

    min_return_threshold = (1 - epsilon) * target_return
    close_to_target = close_to_target[close_to_target['Returns'] > min_return_threshold]

    # Apply the functions to each row
    close_to_target['ENB'] = close_to_target.apply(lambda x: calculate_enb(x, selected_assets, selected_covar, t_mt_cc), axis=1)

    # Find the maximum ENB and associated weights
    max_enb_row = close_to_target.loc[close_to_target['ENB'].idxmax()]
    max_enb = max_enb_row['ENB']
    # Extract and store each asset weight in its own column
    row_data = {'Target Stdev': target_stdev, 'Max ENB': max_enb}
    for asset in selected_assets:
        row_data[asset] = max_enb_row[asset]

    # Store the results
    results_df = results_df.append(row_data, ignore_index=True)

# Optionally, plot the results
plt.figure(figsize=(10, 6))
sns.lineplot(x='Target Stdev', y='Max ENB', data=results_df)
plt.title('Max ENB for Different Target Standard Deviations')
plt.xlabel('Target Standard Deviation')
plt.ylabel('Max ENB')
plt.show()

# results_df now contains the maximum ENB for each target standard deviation
print(results_df)

# Set 'Target Stdev' as the index for plotting
plot_data = results_df.set_index('Target Stdev')[selected_assets]

# Assuming 'color_mapping' is a dictionary mapping asset names to colors
# Convert color mapping to hex format
hex_colors = [rgb_to_hex(color_mapping[asset]) for asset in selected_assets]  # Make sure assets are in the same order as columns

# Create the stacked bar chart
plt.figure(figsize=(10, 6))
ax = plot_data.plot(kind='bar', stacked=True, color=hex_colors)

# Adding titles and labels
plt.title(f'Asset Allocation by Target Volatility on {selected_date}')
plt.xlabel('Target Standard Deviation')
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

################################################################################

import numpy as np

selected_assets = ['Asset 1', 'Asset 2', 'Asset 3', 'Asset 4']

# Define the expected returns for the two assets
expected_return_asset1 = 0.099
expected_return_asset2 = 0.10
expected_return_asset3 = 0.12
expected_return_asset4 = 0.06

# Define the standard deviation for both assets (same for both)
std_dev_asset1 = 0.15
std_dev_asset2 = 0.15
std_dev_asset3 = 0.2
std_dev_asset4 = 0.1

std_devs = np.array([std_dev_asset1, std_dev_asset2, std_dev_asset3, std_dev_asset4])
std_dev_outer_product = np.outer(std_devs, std_devs)

# Define correlation coefficients
correlation_1_2 = 0.95
correlation_1_3 = 0.2
correlation_1_4 = 0.3
correlation_2_3 = 0.2
correlation_2_4 = 0.3
correlation_3_4 = 0.5

# Create the 4x4 correlation matrix
correlation_matrix = np.array([
    [1, correlation_1_2, correlation_1_3, correlation_1_4],
    [correlation_1_2, 1, correlation_2_3, correlation_2_4],
    [correlation_1_3, correlation_2_3, 1, correlation_3_4],
    [correlation_1_4, correlation_2_4, correlation_3_4, 1]
])

covariance_matrix = correlation_matrix * std_dev_outer_product

# Calculate the covariance based on the standard deviation and correlation
# Create covariance matrix

# Create the vector of expected returns
expected_returns = np.array([expected_return_asset1, expected_return_asset2,
                             expected_return_asset3, expected_return_asset4])

min_vals = np.zeros(4)
max_vals = np.ones(4)
w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05 ** (5 / len(min_vals))), target=1)
w = generate_weights(minimum=min_vals, maximum=max_vals, increment=0.01, target=1)

block_arithmetic_return = expected_returns + np.diag(covariance_matrix) / 2

block_result = block_arithmetic_return.dot(w.transpose())

stdevs = []
for w_row in w:
    stdev = np.sqrt(w_row @ covariance_matrix @ w_row.T)  # @ is matrix multiplication
    stdevs.append(stdev)

# Create a DataFrame for returns and stdevs
df = pd.DataFrame({
    "Returns": block_result,
    "Stdevs": stdevs
})

# Add each column of selected_weights to the DataFrame
for i in range(w.shape[1]):
    df[selected_assets[i]] = w[:, i]

t_mt_cc = torsion(covariance_matrix, 'minimum-torsion', method='exact')

max_return = df['Returns'].max()
threshold = 0.9 * max_return
selected_rows = df[df['Returns'] > threshold]

# Apply the functions to each row
selected_rows['ENB'] = selected_rows.apply(lambda x: calculate_enb(x, assets=['Asset 1',
                                                        'Asset 2',
                                                        'Asset 3',
                                                        'Asset 4'],
                                             covariance_matrix=covariance_matrix,
                                             t_mt=t_mt_cc), axis=1)

sorted_df_example = selected_rows.sort_values(by='ENB', ascending=False)

# Plot for ENB
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Stdevs', y='Returns', hue='ENB', data=close_to_target, palette='viridis', alpha=0.6, edgecolor='w')
plt.title('Jitter Plot with ENB')
plt.xlabel('Standard Deviations')
plt.ylabel('Returns')
plt.legend(title='ENB', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot for Diversification Ratio Squared
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Stdevs', y='Returns', hue='Diversification Ratio Squared', data=close_to_target, palette='magma', alpha=0.6, edgecolor='w')
plt.title('Jitter Plot with Diversification Ratio Squared')
plt.xlabel('Standard Deviations')
plt.ylabel('Returns')
plt.legend(title='Diversification Ratio Squared', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

sorted_df = close_to_target.sort_values(by='ENB', ascending=False)
selected_weights_ts_pd = selected_weights_ts_pd.astype(np.float64)
selected_weights_cc_pd = selected_weights_cc_pd.astype(np.float64)
calculate_enb(selected_weights_ts_pd)
calculate_enb(selected_weights_cc_pd)
