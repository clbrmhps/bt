# Standard library imports
import os
import time
import warnings

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import seaborn as sns
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from scipy.optimize import minimize
import bt
import ffn
from plotly.subplots import make_subplots
from tqdm import tqdm  # Import tqdm

from reporting.tools.style import set_clbrm_style
from meucci.EffectiveBets import EffectiveBets
from meucci.torsion import torsion
from analysis.drawdowns import endpoint_mdd_lookup
from ffn.core import calc_erc_weights

import numpy as np
import cvxpy as cp
import os
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize

from meucci.torsion import torsion
from ffn.core import calc_erc_weights

target_stdevs = np.arange(0.05, 0.13, 0.0025)
target_volatility = 0.07
epsilon = 0.1

warnings.simplefilter(action='default', category=RuntimeWarning)

set_clbrm_style(caaf_colors=True)

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

config = 8

plots_dir = f"./plots/config_{config}"
os.makedirs(plots_dir, exist_ok=True)

frontiers_dir = f"./frontiers/config_{config}"
os.makedirs(frontiers_dir, exist_ok=True)

start_date = pd.Timestamp('1875-01-31')
dates = er.index.unique()[er.index.unique() >= start_date]

def rgb_to_hex(rgb_str):
    rgb = rgb_str.replace('rgb', '').replace('(', '').replace(')', '').split(',')
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

COLOR_MAPPING = {
    "Equities": "rgb(64, 75, 151)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Alternatives": "rgb(160, 84, 66)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)"
}

HEX_COLORS = {k: rgb_to_hex(v) for k, v in COLOR_MAPPING.items()}

def diversification_ratio_squared(w, sigma_port, standard_deviations):
    return (np.dot(w, standard_deviations) / sigma_port) ** 2

def calculate_portfolio_properties(caaf_weights, arithmetic_mu, covar):
    aligned_weights, aligned_mu = caaf_weights.align(arithmetic_mu, join='inner')

    portfolio_arithmetic_mu = pf_mu(aligned_weights, aligned_mu)
    portfolio_sigma = pf_sigma(aligned_weights, covar)
    portfolio_geo_mu = portfolio_arithmetic_mu - 0.5 * portfolio_sigma ** 2
    portfolio_md = endpoint_mdd_lookup(portfolio_geo_mu, portfolio_sigma, frequency='M', percentile=5)

    div_ratio_squared = diversification_ratio_squared(aligned_weights, portfolio_sigma, np.sqrt(np.diag(covar)))

    t_mt = torsion(covar, 'minimum-torsion', method='exact')
    p, enb = EffectiveBets(aligned_weights.to_numpy(), covar.to_numpy(), t_mt)

    portfolio_properties = pd.Series({
        'arithmetic_mu': portfolio_arithmetic_mu,
        'sigma': portfolio_sigma,
        'md': portfolio_md,
        'enb': enb[0, 0],
        'div_ratio_sqrd': div_ratio_squared
    })

    return portfolio_properties

def generate_multiplier(n):
    # Convert the number to a string and find the position of the decimal point
    n_str = str(n)
    decimal_pos = n_str.find('.')

    # If there's no decimal, the number is an integer, return 5
    if decimal_pos == -1:
        return 5

    # Find the highest non-zero decimal place
    d = 0
    for digit in n_str[decimal_pos + 1:]:
        if digit != '0':
            break
        d += 1

    # Create a number with a "1" at the highest non-zero decimal place
    new_number = 1 / (10 ** (d + 1))

    # Multiply by 1
    new_number *= 5

    return new_number

def percentage_formatter(x):
    return f"{100 * x:.0f}%"

def filter_rows(group):
    max_arithmetic_mu = group['arithmetic_mu'].max() - 0.001
    sigma_at_max_mu = group.loc[group['arithmetic_mu'].idxmax(), 'sigma']

    return group[(group['arithmetic_mu'] >= max_arithmetic_mu) | (group['sigma'] <= sigma_at_max_mu)]

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


def constraint_sum_to_one(x):
    return np.sum(x) - 1


def volatility_constraint(weights, covar, target_volatility):
    port_vol = np.sqrt(np.dot(np.dot(weights, covar), weights))
    return port_vol - target_volatility

def return_objective(weights, exp_rets):
    mean = sum(exp_rets * weights)
    return mean

def EffectiveBets_scalar(x, Sigma, t_MT):
    _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
    return -scalar_matrix.item()

def pf_mu(weight, mu):
    return np.array([np.inner(weight, mu)])

def pf_sigma(weight, cov):
    weight = weight.ravel()
    return np.sqrt(np.einsum('i,ij,j->', weight, cov, weight))

def EffectiveBets_sc(x, Sigma, t_MT, laglambda, arithmetic_mu):
    _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
    return -scalar_matrix.item() - laglambda * np.matmul(np.transpose(x), arithmetic_mu)

# Define a function to format percentages
def fmt(s):
    try:
        return "{:.1%}".format(float(s))
    except ValueError:
        return ""

# Plotting function for stacked bar chart and efficient frontier
def plot_charts(df, label, frontier_color, plot_filename_suffix, include_enb=False, include_dr=False,
                selected_assets=None, selected_date=None, mv_df=None):
    plt.figure(figsize=(10, 6))
    ax = df.loc[:, selected_assets].plot(kind='bar', stacked=True, color=HEX_COLORS)
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
    #     plt.scatter(combined_df['sigma'], combined_df['arithmetic_mu'], c=combined_df['div_ratio_sqrd'] * 100, cmap='viridis', label='DR')
    # if include_enb:
    #     plt.scatter(combined_df['sigma'], combined_df['arithmetic_mu'], c=combined_df['enb'], cmap='viridis', label='ENB')

    plot_filename = f"{plots_dir}/frontiers_{plot_filename_suffix}.png"
    plt.savefig(plot_filename)
    plt.clf()

def process_date(current_date, er, covar, rdf, frontiers_dir, plots_dir):

    print(f"Processing date: {current_date}")
    formatted_date = current_date.strftime('%Y%m%d')

    print(current_date)

    selected_date = current_date
    selected_assets = list(er.loc[selected_date].dropna().index)

    selected_covar = covar.loc[selected_assets, selected_assets]
    selected_er = er.loc[selected_date, selected_assets]
    stds = np.sqrt(np.diag(selected_covar))
    arithmetic_mu = selected_er + np.square(stds) / 2

    t_mt = torsion(selected_covar, 'minimum-torsion', method='exact')
    x0 = np.full((len(selected_assets),), 1/len(selected_assets))

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

    constraints = ({'type': 'eq', 'fun': constraint_sum_to_one},
                   {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_volatility)})
    bounds = [(0, 1) for _ in range(len(selected_assets))]  # Assuming x is already defined

    ################################################################################################
    # Mean Variance Optimization

    w = cp.Variable(len(arithmetic_mu))

    objective = cp.Minimize(cp.quad_form(w, selected_covar))

    constraint_weight = cp.sum(w) == 1
    constraint_box = w >= 0

    constraints = [constraint_weight, constraint_box]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == cp.OPTIMAL:
        min_var_weights = w.value
        min_var_expected_return = np.sum(min_var_weights * arithmetic_mu)

    target_returns = np.linspace(min_var_expected_return, max(arithmetic_mu), num=1000)

    mv_df = pd.DataFrame()

    for target_return in target_returns:
        w = cp.Variable(len(arithmetic_mu))

        objective = cp.Minimize(cp.quad_form(w, selected_covar))

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

    min_vals = np.zeros(len(selected_assets))
    max_vals = np.ones(len(selected_assets))

    ################################################################################
    # Mean Variance Optimization Scipy

    # mv_df = pd.DataFrame()

    # for target_vol in target_stdevs:
    #     constraints = (
    #         {'type': 'eq', 'fun': constraint_sum_to_one},
    #         {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)}
    #     )

    #     result = minimize(
    #         lambda w: -pf_mu(w, arithmetic_mu),  # We minimize the negative return to maximize the return
    #         x0,
    #         method='SLSQP',
    #         bounds=bounds,
    #         constraints=constraints,
    #         options={'maxiter': 10_000, 'ftol': 1e-15}
    #     )

    #     target_return = -result.fun
    #     mv_df = mv_df.append({'Target Vol': target_vol, 'Target Return': target_return,
    #                         'Portfolio Variance': target_vol**2, 'Portfolio Std Dev': target_vol,
    #                       **dict(zip(selected_assets, result.x))}, ignore_index=True)

    # import pandas as pd
    # import numpy as np
    # from scipy.optimize import minimize


    # def portfolio_variance(w, cov_matrix):
    #     return w.T @ cov_matrix @ w


    # def constraint_sum_to_one(w):
    #     return np.sum(w) - 1

    # def target_return_constraint(w, expected_returns, target_return):
    #     return np.dot(w, expected_returns) - target_return


    # mv_df = pd.DataFrame()
    # x0 = np.array([1 / len(arithmetic_mu)] * len(arithmetic_mu))  # Initial guess
    # bounds = [(0, 1) for _ in range(len(arithmetic_mu))]  # Non-negative weights

    # for target_return in target_returns:
    #     constraints = (
    #         {'type': 'eq', 'fun': constraint_sum_to_one},
    #         {'type': 'eq', 'fun': lambda w: target_return_constraint(w, arithmetic_mu, target_return)}
    #     )

    #     result = minimize(
    #         lambda w: portfolio_variance(w, selected_covar),
    #         x0,
    #         method='SLSQP',
    #         bounds=bounds,
    #         constraints=constraints,
    #         options={'maxiter': 10000, 'ftol': 1e-15}
    #     )

    #     if result.success:
    #         portfolio_variance_var = result.fun
    #         portfolio_std_dev = np.sqrt(portfolio_variance_var)
    #         new_row = {'Target Return': target_return, 'Portfolio Variance': portfolio_variance_var,
    #                    'Portfolio Std Dev': portfolio_std_dev, **dict(zip(selected_assets, result.x))}
    #         mv_df = mv_df.append(new_row, ignore_index=True)


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

    n = len(arithmetic_mu)

    maxenb_weights = pd.Series(x0)
    maxdr_weights = pd.Series(x0)

    results_df = pd.DataFrame(columns=selected_assets)
    results_dr = pd.DataFrame(columns=selected_assets)

    # start_index = len(target_stdevs) // 3
    start_index = 0
    starting_target_vol = target_stdevs[start_index]

    target_stdevs_above = target_stdevs[start_index:]
    target_stdevs_below = target_stdevs[:start_index][::-1]

    penalty_coeff = 40

    if len(target_stdevs_above) > 0:
        for target_vol in target_stdevs_above:
            mv_df['Difference'] = abs(mv_df['Portfolio Std Dev'] - target_vol)
            closest_row = mv_df.iloc[mv_df['Difference'].idxmin()]
            target_return = closest_row['Target Return']

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

                constraints = (
                    {'type': 'eq', 'fun': constraint_sum_to_one},
                    {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                    {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
                )

                result = minimize(
                    enb_soft_constraint_objective,
                    weight_ref,
                    args=(selected_covar.to_numpy(), t_mt, weight_ref, "huber", penalty_coeff),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-10}
                )

                if result.success:
                    maxenb_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
                    aligned_weights, aligned_mu = maxenb_weights.align(arithmetic_mu, join='inner')
                    portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

                    # Add the optimized weights to the DataFrame
                    new_row = pd.DataFrame({'ENB': -result.fun ,'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
                                          **dict(zip(selected_assets, result.x))}, index=[0])
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                ################################################################################################
                # Maximum DR Portfolio

                # def dr_soft_constraint_objective(w, *args):
                #     sigma, standard_deviations, weight_ref, norm, penalty_coeff = args
                #     original_objective = -diversification_ratio(w, sigma, standard_deviations)  # Adjust as necessary
                #     penalty_term = penalty_coeff * weight_objective(w, weight_ref, norm)
                #     return original_objective + penalty_term

                # weight_ref = maxenb_weights.to_numpy()

                # constraints = (
                #     {'type': 'eq', 'fun': constraint_sum_to_one},
                #     {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                #     {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
                # )

                # variances = np.diag(selected_covar)
                # standard_deviations = np.sqrt(variances)

                # result = minimize(
                #     dr_soft_constraint_objective,
                #     x0,
                #     args=(selected_covar.to_numpy(), standard_deviations, weight_ref, "huber", penalty_coeff),
                #     method='SLSQP',
                #     bounds=bounds,
                #     constraints=constraints,
                #     options={'maxiter': 1000, 'ftol': 1e-10}
                # )

                # if result.success:
                #     maxdr_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
                #     aligned_weights, aligned_mu = maxdr_weights.align(arithmetic_mu, join='inner')
                #     portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

                #     # Add the optimized weights to the DataFrame
                #     new_row = pd.DataFrame({'ENB': -result.fun ,'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
                #                           **dict(zip(selected_assets, result.x))}, index=[0])
                #     results_dr = pd.concat([results_dr, new_row], ignore_index=True)

    if len(target_stdevs_below) > 0:
        for target_vol in target_stdevs_below:
            mv_df['Difference'] = abs(mv_df['Portfolio Std Dev'] - target_vol)
            closest_row = mv_df.iloc[mv_df['Difference'].idxmin()]
            target_return = closest_row['Target Return']

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

                constraints = (
                    {'type': 'eq', 'fun': constraint_sum_to_one},
                    {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                    {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
                )

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
                new_row = pd.DataFrame({'ENB': -result.fun ,'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
                                      **dict(zip(selected_assets, result.x))}, index=[0])
                results_df = pd.concat([results_df, new_row], ignore_index=True)

                ################################################################################################
                # Maximum DR Portfolio

                # def dr_soft_constraint_objective(w, *args):
                #     sigma, standard_deviations, weight_ref, norm, penalty_coeff = args
                #     original_objective = -diversification_ratio(w, sigma, standard_deviations)  # Adjust as necessary
                #     penalty_term = penalty_coeff * weight_objective(w, weight_ref, norm)
                #     return original_objective + penalty_term

                # weight_ref = maxenb_weights.to_numpy()

                # constraints = (
                #     {'type': 'eq', 'fun': constraint_sum_to_one},
                #     {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                #     {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
                # )

                # variances = np.diag(selected_covar)
                # standard_deviations = np.sqrt(variances)

                # result = minimize(
                #     dr_soft_constraint_objective,
                #     x0,
                #     args=(selected_covar.to_numpy(), standard_deviations, weight_ref, "huber", penalty_coeff),
                #     method='SLSQP',
                #     bounds=bounds,
                #     constraints=constraints,
                #     options={'maxiter': 1000, 'ftol': 1e-10}
                # )

                # maxdr_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
                # aligned_weights, aligned_mu = maxdr_weights.align(arithmetic_mu, join='inner')
                # portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

                # # Add the optimized weights to the DataFrame
                # new_row = pd.DataFrame({'ENB': -result.fun ,'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
                #                       **dict(zip(selected_assets, result.x))}, index=[0])
                # results_dr = pd.concat([results_dr, new_row], ignore_index=True)

    # Sort the DataFrames by 'Target Vol'
    mv_df = mv_df.sort_values(by='Target Vol')
    results_df = results_df.sort_values(by='Target Vol')  # Ensuring 'Return' is the column name for target return in results_df
    # results_dr = results_dr.sort_values(by='Target Vol')

    results_df.to_pickle(os.path.join(frontiers_dir, f"enb_{current_date.strftime('%Y%m%d')}"))
    # results_dr.to_pickle(os.path.join(frontiers_dir, f"dr_{current_date.strftime('%Y%m%d')}"))

    # Calculate (1 - epsilon) * "Target Return"
    mv_df['Adjusted Return'] = (1 - epsilon) * mv_df['Target Return']

    plot_charts(results_df, 'Max ENB Frontier', 'blue', f'enb_{current_date.date().strftime("%Y%m%d")}', include_enb=include_enb,
                selected_assets=selected_assets, selected_date=selected_date, mv_df=mv_df, combined_df=results_df)

    # Plot for results_dr (assuming 'results_dr' is another DataFrame similar to 'results_df')
    # plot_charts(results_dr, 'Max DR Frontier', 'green', f'dr_{current_date.date().strftime("%Y%m%d")}', include_dr=include_dr)

    return {"date": current_date, "data": "some results"}


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    num_cores = os.cpu_count()
    num_cores = 30

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_date, dates, [er] * len(dates), [covar] * len(dates), [rdf] * len(dates), [frontiers_dir] * len(dates), [plots_dir] * len(dates)))

    print(results)

