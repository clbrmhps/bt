# Standard library imports
import os
import warnings

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import seaborn as sns

import bt
import ffn
from plotly.subplots import make_subplots
from tqdm import tqdm  # Import tqdm
import cvxpy as cp

# Local application/library specific imports
from reporting.tools.style import set_clbrm_style
from meucci.EffectiveBets import EffectiveBets
from meucci.torsion import torsion
from base_portfolio.calculations import pf_mu, pf_sigma
from analysis.drawdowns import endpoint_mdd_lookup
from ffn.core import calc_erc_weights

# Set default runtime warnings behavior
warnings.simplefilter(action='default', category=RuntimeWarning)

# Color mapping from names to RGB strings
color_mapping = {
    "DM Equities": "rgb(64, 75, 151)",
    "EM Equities": "rgb(131, 140, 202)",
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

def diversification_ratio_squared(w, sigma_port, standard_deviations):
    return (np.dot(w, standard_deviations) / sigma_port) ** 2

def diversification_ratio(w, sigma, standard_deviations):
    portfolio_variance = np.dot(w.T, np.dot(sigma, w))
    sigma_port = np.sqrt(portfolio_variance)
    return (np.dot(w, standard_deviations) / sigma_port) ** 2

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

def EffectiveBets_scalar(x, Sigma, t_MT):
    _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
    return -scalar_matrix.item()

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

    div_ratio_squared = diversification_ratio_squared(aligned_weights, portfolio_sigma, np.sqrt(np.diag(covar)))

    t_mt = torsion(covar, 'minimum-torsion', method='exact')
    p, enb = EffectiveBets(aligned_weights.to_numpy(), covar.to_numpy(), t_mt)

    # Compile portfolio properties into a pandas Series
    portfolio_properties = pd.Series({
        'arithmetic_mu': portfolio_arithmetic_mu,
        'sigma': portfolio_sigma,
        'md': portfolio_md,
        'enb': enb[0, 0],
        'div_ratio_sqrd': div_ratio_squared
    })

    return portfolio_properties

def EffectiveBets_sc(x, Sigma, t_MT, laglambda, arithmetic_mu):
    _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
    return -scalar_matrix.item() - laglambda * np.matmul(np.transpose(x), arithmetic_mu)

def constraint_sum_to_one(x):
        return np.sum(x) - 1

class MeanVarianceOptimizer:
    def __init__(self, returns, cov_matrix, type="minimum_variance", method="cvxpy"):
        self.returns = returns
        self.assets = list(self.returns.index)
        self.cov_matrix = cov_matrix
        self.type = type
        self.method = method
        self.n_assets = len(returns)
        self.equal_weight = np.array([1 / self.n_assets] * self.n_assets)

        self.minvar_weights = self.construct_minimum_variance_portfolio()
        self.minvar_return = self.calculate_portfolio_return(self.minvar_weights)
        self.minvar_std_dev = np.sqrt(self.calculate_portfolio_variance(self.minvar_weights))

        self.target_returns = np.linspace(self.minvar_return, max(returns))
        # self.target_std_devs = np.linspace(self.minvar_std_dev, max(np.sqrt(np.diag(self.cov_matrix))))
        self.target_std_devs = np.arange(0.05, 0.13, 0.0025)  # Adjust this range as needed

    def calculate_portfolio_variance(self, weights):
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    def calculate_portfolio_return(self, weights):
        return np.dot(weights.T, self.returns)

    def construct_minimum_variance_portfolio(self):
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        ones = np.ones(self.n_assets)
        weights = np.dot(inv_cov_matrix, ones) / np.dot(ones.T, np.dot(inv_cov_matrix, ones))
        return weights

    def construct_mean_variance_portfolio(self, target_return=None, target_std_dev=None):
        if self.method == "cvxpy":
            w = cp.Variable(self.n_assets)

            # General constraints
            constraint_weight = cp.sum(w) == 1
            constraint_box = w >= 0

            if self.type == "minimum_variance":
                # Objective
                objective = cp.Minimize(cp.quad_form(w, self.cov_matrix))

                # Constraints
                constraint_return = cp.sum(cp.multiply(w, self.returns)) == target_return
                constraints = [constraint_weight, constraint_return, constraint_box]

                try:
                    problem = cp.Problem(objective, constraints)
                    problem.solve()
                except:
                    print('Error in optimization')

                return w.value if problem.status == 'optimal' else None
        if self.method == "scipy":
            # General constraints
            bounds = [(0, 1) for _ in range(self.n_assets)]
            sum_to_one_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

            if self.type == "minimum_variance":
                # Constraints
                constraints = (
                      sum_to_one_constraint,
                      {'type': 'eq', 'fun': lambda w: self.calculate_portfolio_return(w) - target_return}
                )

                result = minimize(
                            lambda w: self.calculate_portfolio_variance(w),
                            self.equal_weight,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 10000, 'ftol': 1e-15}
                )

                return result.x
            elif self.type == "maximum_return":
                # Constraints
                constraints = (
                      sum_to_one_constraint,
                      {'type': 'eq', 'fun': lambda w: np.sqrt(self.calculate_portfolio_variance(w)) - target_std_dev}
                )

                result = minimize(
                            lambda w: -self.calculate_portfolio_return(w),
                            self.equal_weight,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-15}
                )

                return result.x

    def add_row_to_frontier(self, weights, target_value, df_frontier):
        if weights is not None:
            portfolio_return = self.calculate_portfolio_return(weights)
            portfolio_variance = self.calculate_portfolio_variance(weights)
            portfolio_std_dev = np.sqrt(portfolio_variance)
            new_row = {
                'Portfolio Return': portfolio_return,
                'Portfolio Variance': portfolio_variance,
                'Portfolio Std Dev': portfolio_std_dev,
                'Target Value': target_value,
                'Target Type': self.type,
                **dict(zip(self.assets, weights))
            }
            return pd.concat([df_frontier, pd.DataFrame([new_row])], ignore_index=True)

    def construct_frontier(self):
        df_frontier = pd.DataFrame([])
        targets = self.target_returns if self.type == "minimum_variance" else self.target_std_devs
        for target_value in targets:
            if self.type == "minimum_variance":
                weights = self.construct_mean_variance_portfolio(target_return=target_value)
            elif self.type == "maximum_return":
                weights = self.construct_mean_variance_portfolio(target_std_dev=target_value)

            df_frontier = self.add_row_to_frontier(weights, target_value, df_frontier)
        return df_frontier

def plot_efficient_frontiers(cvxpy_returns, cvxpy_stdevs, min_var_returns, min_var_stdevs, max_ret_returns, max_ret_stdevs, figsize=(10, 6)):
    # Plot setup
    plt.figure(figsize=figsize)

    # Plot each efficient frontier
    plt.plot(cvxpy_stdevs, cvxpy_returns, marker='o', label='Minimum Variance (CVXPY)')
    plt.plot(min_var_stdevs, min_var_returns, marker='x', label='Minimum Variance (SciPy)')
    plt.plot(max_ret_stdevs, max_ret_returns, marker='^', label='Maximum Return (SciPy)')

    # Add labels and legend
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Portfolio Return')
    plt.title('Efficient Frontiers')
    plt.legend()

    # Display plot
    plt.show()

################################################################################
# Configuration

set_clbrm_style(caaf_colors=True)

country = "US"

plots_dir = "./plots"
os.makedirs(plots_dir, exist_ok=True)

################################################################################
# Data Import

# Return data for the covariance matrix
rdf = pd.read_excel(f"../data/master_file.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf.set_index('Date', inplace=True)
# rdf.dropna(inplace=True)

# Expected return data
er = pd.read_excel(f"../data/master_file.xlsx", sheet_name="expected_gross_return")
er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
er.set_index('Date', inplace=True)
er.loc[:"1973-01-31", "Gold"] = np.nan

# Construct and scale covariance matrix
const_covar = rdf.cov()
covar = const_covar * 12

################################################################################
# Strategy Configuration

start_date = pd.Timestamp('2020-01-31')
dates = er.index.unique()[er.index.unique() >= start_date]

target_stdevs = np.arange(0.05, 0.13, 0.0025)  # Adjust this range as needed
target_volatility = 0.07
epsilon = 0.1

################################################################################
# Date Iteration

for current_date in dates:
    formatted_date = current_date.strftime('%Y%m%d')
    print(current_date)

    selected_date = current_date
    selected_assets = list(er.loc[selected_date].dropna().index)
    selected_assets = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds',
                       'Gold', 'Alternatives']

    selected_covar = covar.loc[selected_assets, selected_assets]
    selected_er = er.loc[selected_date, selected_assets]
    selected_stds = np.sqrt(np.diag(selected_covar))
    arithmetic_mu = selected_er + np.square(selected_stds) / 2

    t_mt = torsion(selected_covar, 'minimum-torsion', method='exact')
    x0 = np.full((len(selected_assets),), 1/len(selected_assets))

    ############################################################################
    # ERC Weights

    erc_weights = calc_erc_weights(returns=rdf,
                                   initial_weights=None,
                                   risk_weights=None,
                                   covar_method="constant",
                                   risk_parity_method="slsqp",
                                   maximum_iterations=1000,
                                   tolerance=1e-10,
                                   const_covar=covar.loc[selected_assets, selected_assets])

    ################################################################################################
    # Mean Variance Optimization

    optimizer = MeanVarianceOptimizer(arithmetic_mu, selected_covar, type="minimum_variance", method="cvxpy")
    tmp_cvxpy = optimizer.construct_frontier()

    optimizer = MeanVarianceOptimizer(arithmetic_mu, selected_covar, type="minimum_variance", method="scipy")
    tmp_min_var_scipy = optimizer.construct_frontier()

    optimizer = MeanVarianceOptimizer(arithmetic_mu, selected_covar, type="maximum_return", method="scipy")
    tmp_max_ret_scipy = optimizer.construct_frontier()

    mv_df = tmp_max_ret_scipy

    cvxpy_returns = tmp_cvxpy['Portfolio Return']
    cvxpy_stdevs = tmp_cvxpy['Portfolio Std Dev']
    min_var_returns = tmp_min_var_scipy['Portfolio Return']
    min_var_stdevs = tmp_min_var_scipy['Portfolio Std Dev']
    max_ret_returns = tmp_max_ret_scipy['Portfolio Return']
    max_ret_stdevs = tmp_max_ret_scipy['Portfolio Std Dev']

    plot_efficient_frontiers(cvxpy_returns, cvxpy_stdevs, min_var_returns, min_var_stdevs, max_ret_returns, max_ret_stdevs)

    ################################################################################################
    # Weight Grid

    include_enb = False
    include_dr = False

    # generated_weights = generate_weights(minimum=min_vals, maximum=max_vals, increment=0.05, target=1)
    # weights_df = pd.DataFrame(generated_weights, columns=selected_assets)
    # portfolio_properties_df = pd.DataFrame()

    # for index, row in tqdm(weights_df.iterrows(), total=len(weights_df)):
    #     portfolio_properties = calculate_portfolio_properties(row, arithmetic_mu, selected_covar)
    #     portfolio_properties_df = pd.concat([portfolio_properties_df, pd.DataFrame([portfolio_properties])], ignore_index=True)

    # # Optional: Concatenate the weights and properties DataFrames
    # combined_df = pd.concat([weights_df, portfolio_properties_df], axis=1)

    # include_enb = True
    # include_dr = True

    ################################################################################################
    # Max ENB/DR
    import math

    n = len(arithmetic_mu)
    bounds = [(0, 1) for _ in range(len(selected_assets))]

    maxenb_weights = pd.Series(x0)
    maxdr_weights = pd.Series(x0)

    results_df = pd.DataFrame(columns=selected_assets)
    results_dr = pd.DataFrame(columns=selected_assets)

    for target_vol in target_stdevs:
        mv_df['Difference'] = abs(mv_df['Portfolio Std Dev'] - target_vol)
        closest_row = mv_df.iloc[mv_df['Difference'].idxmin()]
        target_return = closest_row['Portfolio Return']

        if math.isfinite(target_return):
            new_row = pd.DataFrame({'Target Vol': target_vol, 'Target Return': target_return,
                                    **dict(zip(selected_assets, closest_row[selected_assets]))}, index=[0])
            mv_df = pd.concat([mv_df, new_row], ignore_index=True)

            ################################################################################################
            # Maximum ENB Portfolio
            def enb_soft_constraint_objective(w, *args):
                # sigma, t_MT, weight_ref, norm, penalty_coeff = args
                sigma, t_MT, weight_ref, norm, penalty_coeff = args
                original_objective = EffectiveBets_scalar(w, sigma, t_MT)
                penalty_term = penalty_coeff * weight_objective(w, weight_ref, norm)
                return original_objective + penalty_term


            weight_ref = maxenb_weights.to_numpy()
            penalty_coeff = 50

            constraints = (
                {'type': 'eq', 'fun': constraint_sum_to_one},
                {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
            )

            # x0 = closest_row.loc[['Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']]

            result = minimize(
                enb_soft_constraint_objective,
                x0,
                args=(selected_covar.to_numpy(), t_mt, weight_ref, "huber", penalty_coeff),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )

            maxenb_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
            aligned_weights, aligned_mu = maxenb_weights.align(arithmetic_mu, join='inner')
            portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

            # Add the optimized weights to the DataFrame
            new_row = pd.DataFrame({'ENB': -result.fun, 'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
                                    **dict(zip(selected_assets, result.x))}, index=[0])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            ################################################################################################
            # Maximum DR Portfolio

            def dr_soft_constraint_objective(w, *args):
                sigma, standard_deviations, weight_ref, norm, penalty_coeff = args
                original_objective = -diversification_ratio(w, sigma, standard_deviations)  # Adjust as necessary
                penalty_term = penalty_coeff * weight_objective(w, weight_ref, norm)
                return original_objective + penalty_term


            weight_ref = maxenb_weights.to_numpy()
            penalty_coeff = 0

            constraints = (
                {'type': 'eq', 'fun': constraint_sum_to_one},
                {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
            )

            variances = np.diag(selected_covar)
            standard_deviations = np.sqrt(variances)

            result = minimize(
                dr_soft_constraint_objective,
                x0,
                args=(selected_covar.to_numpy(), standard_deviations, weight_ref, "huber", penalty_coeff),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )

            maxdr_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
            aligned_weights, aligned_mu = maxdr_weights.align(arithmetic_mu, join='inner')
            portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

            # Add the optimized weights to the DataFrame
            new_row = pd.DataFrame({'ENB': -result.fun, 'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
                                    **dict(zip(selected_assets, result.x))}, index=[0])
            results_dr = pd.concat([results_dr, new_row], ignore_index=True)

    # Sort the DataFrames by 'Target Vol'
    mv_df = mv_df.sort_values(by='Target Vol')
    results_df = results_df.sort_values(by='Target Vol')  # Ensuring 'Return' is the column name for target return in results_df

    # Calculate (1 - epsilon) * "Target Return"
    mv_df['Adjusted Return'] = (1 - epsilon) * mv_df['Target Return']

    # Define a function to format percentages
    def fmt(s):
        try:
            return "{:.1%}".format(float(s))
        except ValueError:
            return ""

    # Plotting function for stacked bar chart and efficient frontier
    def plot_charts(df, label, frontier_color, plot_filename_suffix, include_enb=False, include_dr=False):
        # Plot the stacked bar chart
        plt.figure(figsize=(10, 6))
        ax = df.loc[:, selected_assets].plot(kind='bar', stacked=True, color=hex_colors)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.title(f'Asset Allocation by Target Volatility on {selected_date}')
        plt.xlabel('Target Volatility (Standard Deviation)')
        plt.ylabel('Weight')

        target_vols = df['Target Vol'].apply(fmt).tolist()
        ax.set_xticks(range(len(target_vols)))
        ax.set_xticklabels(target_vols)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plot_filename = f"{plots_dir}/weight_distribution_{plot_filename_suffix}.png"
        plt.savefig(plot_filename)
        plt.clf()

        # Plot the efficient frontier
        plt.figure(figsize=(10, 6))
        plt.plot(mv_df['Target Vol'], mv_df['Target Return'], marker='o', label="MV Frontier")
        plt.plot(df['Sigma'], df['Return'], marker='x', linestyle='--', label=label)
        plt.plot(mv_df['Target Vol'], mv_df['Adjusted Return'], linestyle='-.', color=frontier_color, label=f'(1 - {epsilon}) x MV Frontier')
        plt.title('Comparison of Efficient Frontiers with Adjusted Return')
        plt.xlabel('Target Volatility (Standard Deviation)')
        plt.ylabel('Target Return')
        plt.grid(True)
        plt.legend()

        # if include_dr:
        #    plt.scatter(combined_df['sigma'], combined_df['arithmetic_mu'], c=combined_df['div_ratio_sqrd'] * 100, cmap='viridis', label='DR')
        # if include_enb:
        #    plt.scatter(combined_df['sigma'], combined_df['arithmetic_mu'], c=combined_df['enb'], cmap='viridis', label='ENB')

        plot_filename = f"{plots_dir}/frontiers_{plot_filename_suffix}.png"
        plt.savefig(plot_filename)
        plt.clf()


    # Plot for results_df
    plot_charts(results_df, 'Max ENB Frontier', 'blue', f'enb_{current_date.date().strftime("%Y%m%d")}', include_enb=include_enb)

    # Plot for results_dr (assuming 'results_dr' is another DataFrame similar to 'results_df')
    plot_charts(results_dr, 'Max DR Frontier', 'green', f'dr_{current_date.date().strftime("%Y%m%d")}', include_dr=include_dr)
