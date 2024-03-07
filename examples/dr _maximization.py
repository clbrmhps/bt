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

from base_portfolio.calculations import pf_mu
from base_portfolio.calculations import pf_sigma
from analysis.drawdowns import endpoint_mdd_lookup

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

def diversification_ratio(w, sigma_port, standard_deviations):
    return (np.dot(w, standard_deviations) / sigma_port)

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

# Now call the function with er_aligned, const_covar, and generate_weights
# data_results = process_data_frame(er_aligned, const_covar, generate_weights_function)

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

import matplotlib.pyplot as plt
import os

# Create a directory for plots if it doesn't exist
plots_dir = "./plots_dr"
os.makedirs(plots_dir, exist_ok=True)

# Define the start date
start_date = pd.Timestamp('1985-10-31')

# Filter the DataFrame to include only dates starting from 'start_date'
dates = er.index.unique()[er.index.unique() >= start_date]

# Iterate over each date in the index of er
for current_date in dates:
    # Replace 'selected_date' with 'current_date' in your code
    # For example:
    formatted_date = current_date.strftime('%Y%m%d')

    print(current_date)

    selected_date = current_date
    selected_assets = list(er.loc[selected_date].dropna().index)

    selected_covar = covar.loc[selected_assets, selected_assets]
    selected_er = er.loc[selected_date, selected_assets]
    stds = np.sqrt(np.diag(selected_covar))
    arithmetic_mu = selected_er + np.square(stds) / 2

    erc_weights = calc_erc_weights(returns=rdf,
                                   initial_weights=None,
                                   risk_weights=None,
                                   covar_method="constant",
                                   risk_parity_method="slsqp",
                                   maximum_iterations=1000,
                                   tolerance=1e-10,
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

    diversification_ratio_cc = diversification_ratio(selected_weights_cc, sigma_port_cc, stds)
    diversification_ratio_ts = diversification_ratio(selected_weights_ts, sigma_port_ts, stds)

    Sigma = selected_covar
    t_MT = t_mt_cc

    target_volatility = 0.07

    def constraint_sum_to_one(x):
        return np.sum(x) - 1  # The constraint sum of x_i should be 1
    def volatility_constraint(weights, covar, target_volatility):
        # portfolio volatility
        port_vol = np.sqrt(np.dot(np.dot(weights, covar), weights))
        # we want to ensure our portfolio volatility is equal to the given number
        return port_vol - target_volatility

    def return_objective(weights, exp_rets):
        # portfolio mean
        mean = sum(exp_rets * weights)
        # negative because we want to maximize the portfolio mean
        # and the optimizer minimizes metric
        return mean

    epsilon =  0.1

    # Define the constraints and bounds
    constraints = ({'type': 'eq', 'fun': constraint_sum_to_one},
                   {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_volatility)})
    bounds = [(0, 1) for _ in range(len(selected_assets))]  # Assuming x is already defined

    # Initial guess for x
    x0 = np.full((len(selected_assets),), 1/len(selected_assets))
    def EffectiveBets_scalar(x, Sigma, t_MT):
        _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
        return -scalar_matrix.item()

    EffectiveBets_scalar(x0, Sigma.to_numpy(), t_MT)

    # Perform the optimization
    result = minimize(
        EffectiveBets_scalar,
        x0,
        args=(Sigma.to_numpy(), t_MT),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    x_star = result.x

    ####################################################################################################

    results_df = pd.DataFrame(columns=selected_assets)
    results_ts = pd.DataFrame(columns=selected_assets)
    results_te = pd.DataFrame(columns=selected_assets)
    target_stdevs = np.arange(0.05, 0.13, 0.0025)  # Adjust this range as needed

    mv_df = pd.DataFrame(columns=['Target Vol', 'Target Return'] + selected_assets)
    def pf_mu(weight, mu):
        # if weight.shape != mu.shape:
        #    raise Exception('Variables weight and mu must have same shape.')
        # weight = weight.reshape(-1, 1)
        # mu = mu.reshape(-1, 1)
        # return np.dot(weight.T, mu).ravel()
        return np.array([np.inner(weight, mu)])

    def pf_sigma(weight, cov):
        # assert np.ndim(cov) == 2, 'Covariance matrix has to be 2-dimensional.'
        # assert np.max(weight.shape) == cov.shape[0] == cov.shape[1], 'Shapes of weight and cov are not aligned.'
        # sanity_checks.positive_semidefinite(matrix=cov)
        # weight = weight.reshape(-1, 1)
        # return np.sqrt(np.dot(weight.T, np.dot(cov, weight))).ravel()
        weight = weight.ravel()  # Flattening the weight array to 1D
        return np.sqrt(np.einsum('i,ij,j->', weight, cov, weight))

    def calculate_portfolio_properties(caaf_weights, arithmetic_mu, covar):
        # Align weights and arithmetic mean returns
        aligned_weights, aligned_mu = caaf_weights.align(arithmetic_mu, join='inner')

        # Calculate portfolio metrics
        portfolio_arithmetic_mu = pf_mu(aligned_weights, aligned_mu)
        portfolio_sigma = pf_sigma(aligned_weights, covar)
        portfolio_geo_mu = portfolio_arithmetic_mu - 0.5 * portfolio_sigma ** 2
        portfolio_md = endpoint_mdd_lookup(portfolio_geo_mu, portfolio_sigma, frequency='M', percentile=5)

        div_ratio = diversification_ratio(aligned_weights, portfolio_sigma, np.sqrt(np.diag(covar)))

        t_mt = torsion(covar, 'minimum-torsion', method='exact')
        p, enb = EffectiveBets(aligned_weights.to_numpy(), covar.to_numpy(), t_mt)

        # Compile portfolio properties into a pandas Series
        portfolio_properties = pd.Series({
            'arithmetic_mu': portfolio_arithmetic_mu,
            'sigma': portfolio_sigma,
            'md': portfolio_md,
            'enb': enb[0, 0],
            'div_ratio': div_ratio
        })

        return portfolio_properties

    ################################################################################################
    # Block Generation

    min_vals = np.zeros(len(selected_assets))
    max_vals = np.ones(len(selected_assets))
    generated_weights = generate_weights(minimum=min_vals, maximum=max_vals, increment=0.05, target=1)

    weights_df = pd.DataFrame(generated_weights, columns=selected_assets)

    portfolio_properties_df = pd.DataFrame()

    for index, row in weights_df.iterrows():
        portfolio_properties = calculate_portfolio_properties(row, arithmetic_mu, selected_covar)
        portfolio_properties_df = pd.concat([portfolio_properties_df, pd.DataFrame([portfolio_properties])], ignore_index=True)

    # Optional: Concatenate the weights and properties DataFrames
    combined_df = pd.concat([weights_df, portfolio_properties_df], axis=1)

    ################################################################################################
    # Mean Variance Optimization

    import cvxpy as cp
    import numpy as np
    import math

    # Initialize variables for optimization
    w = cp.Variable(len(arithmetic_mu))

    # Objective: Minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(w, selected_covar))

    # Constraints
    constraint_weight = cp.sum(w) == 1
    constraint_box = w >= 0

    constraints = [constraint_weight, constraint_box]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Check if the problem is solved successfully
    if problem.status == cp.OPTIMAL:
        # Portfolio weights for minimum variance
        min_var_weights = w.value

        # Calculate expected return for minimum variance portfolio
        min_var_expected_return = np.sum(min_var_weights * arithmetic_mu)

        print("Minimum Variance Portfolio Expected Return:", min_var_expected_return)

    # Create a list of target returns to iterate over
    target_returns = np.linspace(min_var_expected_return, max(arithmetic_mu), num=20)

    # Initialize an empty DataFrame to store the results
    mv_df = pd.DataFrame()

    for target_return in target_returns:

        w = cp.Variable(len(arithmetic_mu))

        # Objective: Minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(w, selected_covar))

        # Constraints
        constraint_weight = cp.sum(w) == 1
        constraint_return = cp.sum(cp.multiply(w, arithmetic_mu)) == target_return
        constraint_box = w >= 0

        constraints = [constraint_weight, constraint_return, constraint_box]

        try:
            problem = cp.Problem(objective, constraints)
            problem.solve()
        except:
            print('Error in optimization')
            continue

        # Record the results if the problem is solved successfully
        if problem.status == cp.OPTIMAL:
            portfolio_variance = problem.value
            portfolio_std_dev = math.sqrt(portfolio_variance)

            new_row = pd.DataFrame({
                'Target Return': target_return,
                'Portfolio Variance': portfolio_variance,
                'Portfolio Std Dev': portfolio_std_dev,
                **dict(zip(selected_assets, w.value))
            }, index=[0])
            mv_df = pd.concat([mv_df, new_row], ignore_index=True)

            ################################################################################################
            # Maximum Diversification Portfolio

            n = Sigma.shape[0]
            sigma_i = np.sqrt(np.diag(Sigma))

            # Define the optimization variable for weights
            y = cp.Variable(n)

            # The objective function is to minimize y^T * Sigma * y, which is the portfolio variance
            objective = cp.Minimize(cp.quad_form(y, Sigma))
            constraint_std = cp.sum(cp.multiply(y, sigma_i)) == 1
            constraint_return = cp.sum(cp.multiply(arithmetic_mu, y)) == target_return * cp.sum(y)
            constraint_box = y >= 0

            constraints = [constraint_std, constraint_return, constraint_box]

            try:
                problem = cp.Problem(objective, constraints)
                problem.solve()

                if problem.status == 'infeasible_inaccurate':
                    print('Infeasible and inaccurate')
                    continue
            except:
                print('Error in optimization')
                continue

            result = y.value / np.sum(y.value)

            maxdr_weights = pd.Series({arithmetic_mu.index[i]: result[i] for i in range(n)})
            aligned_weights, aligned_mu = maxdr_weights.align(arithmetic_mu, join='inner')
            portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

            # Add the optimized weights to the DataFrame
            new_row = pd.DataFrame({'ENB': portfolio_properties['enb'], 'Target Return': target_return, 'Sigma': portfolio_properties['sigma'],
                                  **dict(zip(selected_assets, result))}, index=[0])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

    n = len(arithmetic_mu)

    mv_df.plot(kind='scatter', x='Portfolio Std Dev', y='Target Return')
    plt.title('Portfolio Std Dev vs Target Return')
    plt.show()

    # results_df.plot(kind='scatter', x='Sigma', y='Target Return')
    # plt.title('Portfolio Std Dev vs Target Return')
    # plt.show()

    def EffectiveBets_sc(x, Sigma, t_MT, laglambda, arithmetic_mu):
        _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
        return -scalar_matrix.item() - laglambda * np.matmul(np.transpose(x), arithmetic_mu)

    # for laglambda in np.arange(0, 200, 5):
    #     ###############################################################################################
    #     # Soft Constraint
    #     constraints = (
    #         {'type': 'eq', 'fun': constraint_sum_to_one},
    #     )

    #     result = minimize(
    #         EffectiveBets_sc,
    #         x0,
    #         args=(Sigma.to_numpy(), t_MT, laglambda, arithmetic_mu.to_numpy()),
    #         method='SLSQP',
    #         bounds=bounds,
    #         constraints=constraints,
    #         options={'maxiter': 1000, 'ftol': 1e-10}
    #     )

    #     sc_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
    #     aligned_weights, aligned_mu = sc_weights.align(arithmetic_mu, join='inner')
    #     portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

    #     # Add the optimized weights to the DataFrame
    #     results_sc = results_sc.append({'ENB': portfolio_properties['enb'],  'Lambda': laglambda, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
    #                           **dict(zip(selected_assets, result.x))}, ignore_index=True)

    # Sort the DataFrames by 'Target Vol'
    mv_df = mv_df.sort_values(by='Target Return')
    results_df = results_df.sort_values(by='Target Return')  # Make sure 'Return' is the column name for target return in results_df
    # Calculate (1 - epsilon) * "Target Return"
    mv_df['Adjusted Return'] = (1 - epsilon) * mv_df['Target Return']

    ####################################################################################################
    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))
    ax = results_df.loc[:, selected_assets].plot(kind='bar', stacked=True, color=hex_colors)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))  # Format the y-axis labels as percentages
    plt.title(f'Asset Allocation by Target Volatility on {selected_date}')
    plt.xlabel('Target Volatility (Standard Deviation)')
    plt.ylabel('Weight')
    def fmt(s):
        try:
            n = "{:.1%}".format(float(s))
        except:
            n = ""
        return n

    # Assuming 'Target Vol' column is not the index and needs to be used as the x-axis labels
    target_vols = results_df['Sigma'].apply(fmt).tolist()  # Apply the formatting function and convert to list

    # Set the x-ticks and labels
    ax.set_xticks(range(len(target_vols)))  # Set x-ticks positions
    ax.set_xticklabels(target_vols)  # Set x-tick labels

    # Position the legend to the right of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot with adjustments
    plt.tight_layout()  # Adjust the layout
    # Save the plot
    plot_filename = f"{plots_dir}/weight_distribution_{formatted_date}.png"
    plt.savefig(plot_filename)

    # Clear the current figure after saving (to avoid overlap in the next iteration)
    plt.clf()

    # Start the plot
    plt.figure(figsize=(10, 6))

    # Plot the efficient frontier for mv_df
    plt.plot(mv_df['Portfolio Std Dev'], mv_df['Target Return'], marker='o', label="MV Frontier")

    # Plot the efficient frontier for results_df
    plt.plot(results_df['Sigma'], results_df['Target Return'], marker='x', linestyle='--', label='Max DR Frontier')

    # Plot the (1 - epsilon) * "Target Return" line
    plt.plot(mv_df['Portfolio Std Dev'], mv_df['Adjusted Return'], linestyle='-.', color='red', label=f'(1 - {epsilon}) x MV Frontier')

    # Add title, labels, grid, legend, and show the plot
    plt.title('Comparison of Efficient Frontiers with Adjusted Return')
    plt.xlabel('Target Volatility (Standard Deviation)')
    plt.ylabel('Target Return')
    plt.grid(True)
    plt.legend()  # This adds the legend to distinguish the different lines
    # Save the plot
    plot_filename = f"{plots_dir}/frontiers_{formatted_date}.png"
    plt.savefig(plot_filename)

    # Clear the current figure after saving (to avoid overlap in the next iteration)
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(mv_df['Portfolio Std Dev'], mv_df['Target Return'], marker='o', label="MV Frontier")
    plt.plot(results_df['Sigma'], results_df['Target Return'], marker='x', linestyle='--', label='Max DR Frontier')
    plt.plot(mv_df['Portfolio Std Dev'], mv_df['Adjusted Return'], linestyle='-.', color='red', label=f'(1 - {epsilon}) x MV Frontier')

    # Scatter plot for your portfolio properties
    # Assuming portfolio_properties_df is your DataFrame with the required columns
    plt.scatter(portfolio_properties_df['sigma'], portfolio_properties_df['arithmetic_mu'],
                c=portfolio_properties_df['div_ratio'], cmap='viridis', label='Portfolio Points')

    # Add colormap legend for div_ratio
    plt.colorbar(label='Diversification Ratio')

    # Finalizing the plot
    plt.title('Comparison of Efficient Frontiers with Adjusted Return')
    plt.xlabel('Target Volatility (Standard Deviation)')
    plt.ylabel('Target Return')
    plt.grid(True)
    plt.legend()
    plt.show()


