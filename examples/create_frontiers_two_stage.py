# Standard library imports
import datetime
import warnings

from reporting.tools.style import set_clbrm_style
from meucci.EffectiveBets import EffectiveBets
from analysis.drawdowns import endpoint_mdd_lookup
from portfolio_construction import calculations, optimization

import numpy as np
import cvxpy as cp
import os
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

from meucci.torsion import torsion

from typing import List, Dict
from ffn.core import add_additional_constraints

from examples.tools import read_benchmark_returns
from examples.effective_rank_calculator import EffectiveRankCalculator

warnings.simplefilter(action='default', category=RuntimeWarning)
set_clbrm_style(caaf_colors=True)

target_stdevs = np.arange(0.01, 0.13, 0.0025)

additional_constraints = {'alternatives_upper_bound': 0.144,
                          #'gold_upper_bound': 0.1,
                          'em_equities_upper_bound': 0.3,
                          'hy_credit_upper_bound': 0.086,
                          }
# additional_constraints = {'alternatives_upper_bound': 0.20,
#                           'em_equities_upper_bound': 0.3,
#                           'hy_credit_upper_bound': 0.30,
#                           'gold_upper_bound': 0.15}
configs_to_run = [29]

def rgb_to_hex(rgb_str):
    rgb = rgb_str.replace('rgb', '').replace('(', '').replace(')', '').split(',')
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

COLOR_MAPPING = {
    "Equities": "rgb(64, 75, 151)",
    "DM Equities": "rgb(64, 75, 151)",
    "EM Equities": "rgb(131, 140, 202)",
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

    t_pca = torsion(covar, 'pca', method='exact')
    p_pca, enb_pca = EffectiveBets(aligned_weights.to_numpy(), covar.to_numpy(), t_pca)

    herfindahl_index = np.sum(np.square(aligned_weights))

    marginal_contribution = np.dot(covar, aligned_weights) / portfolio_sigma
    risk_contributions = np.multiply(marginal_contribution, aligned_weights)
    normalized_rc = risk_contributions / np.sum(risk_contributions)
    herfindahl_index_rc = np.sum(np.square(normalized_rc))

    portfolio_properties = pd.Series({
        'arithmetic_mu': portfolio_arithmetic_mu,
        'sigma': portfolio_sigma,
        'md': portfolio_md,
        'enb': enb[0, 0],
        'div_ratio_sqrd': div_ratio_squared,
        'enb_pca': enb_pca[0, 0],
        'herfindahl_index': herfindahl_index,
        'herfindahl_index_rc': herfindahl_index_rc
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

def fmt(s):
    try:
        return "{:.1%}".format(float(s))
    except ValueError:
        return ""

def plot_charts(df, label, frontier_color, plot_filename_suffix, include_enb=False, include_dr=False,
                selected_assets=None, selected_date=None, mv_df=None, plots_dir=None, config=None, epsilon=None):
    df_downsampled = df.iloc[::(len(df) // 50 or 1)]

    plt.figure(figsize=(10, 6))
    ax = df_downsampled.loc[:, selected_assets].plot(kind='bar', stacked=True, color=HEX_COLORS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.title(f'Asset Allocation by Target Volatility on {selected_date}')
    plt.xlabel('Target Volatility (Standard Deviation)')
    plt.ylabel('Weight')

    target_vols = df_downsampled['Target Vol'].apply(fmt).tolist()
    ax.set_xticks(range(len(target_vols)))
    ax.set_xticklabels(target_vols)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_filename = f"{plots_dir}/weight_distribution_{plot_filename_suffix}_{config['Config']}.png"
    plt.savefig(plot_filename)
    plt.clf()

    if label == 'Max ENB Frontier':
        label = 'CAAF 2.0'

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

    plot_filename = f"{plots_dir}/frontiers_{plot_filename_suffix}_{config['Config']}.png"
    plt.savefig(plot_filename)
    plt.clf()

def get_selected_assets(er: pd.DataFrame, current_date: pd.Timestamp) -> List[str]:
    return list(er.loc[current_date].dropna().index)

def perform_mean_variance_optimization(arithmetic_mu, selected_covar, selected_assets, country,
                                       additional_constraints, tracking_error_limit, rdf, benchmarks):

    calculator = EffectiveRankCalculator(algorithm=None,
                                         minRank=1,
                                         tickers=[],
                                         period=252 * 2,
                                         pctChangePeriod=21,
                                         covar_method='',
                                         resolution=None)

    benchmark_returns = benchmarks['Flex'].pct_change().dropna()

    pdf = 100 * np.cumprod(1 + rdf)
    calculator.calculate(prices=pdf.loc[:, selected_assets])

    combined_returns = rdf.copy()
    combined_returns['Flex'] = benchmarks['Flex'].pct_change().dropna()

    combined_returns = combined_returns.dropna()
    # combined_returns = combined_returns.loc[combined_returns.index <= selected_date]
    cov_matrix = combined_returns.cov() * 12
    asset_benchmark_covar = cov_matrix.loc[selected_assets, 'Flex']
    benchmark_var = np.var(benchmark_returns) * 12

    n = len(arithmetic_mu)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, selected_covar))

    constraint_weight = cp.sum(w) == 1
    constraint_box = w >= 0

    constraints = [constraint_weight, constraint_box]
    constraints = add_additional_constraints(constraints, additional_constraints,
                                             library='cvxpy', country=country, cvxpy_w=w,
                                             number_of_assets=n)

    if tracking_error_limit is not None:
        constraints += [tracking_error_constraint_cvxpy(w, selected_covar, asset_benchmark_covar, benchmark_var, tracking_error_limit)]

    problem = cp.Problem(objective, constraints)
    problem.solve(qcp=True)

    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
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
        constraints = add_additional_constraints(constraints, additional_constraints,
                                                 library='cvxpy', country=country, cvxpy_w=w,
                                                 number_of_assets=n)

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
                'Target Vol': portfolio_std_dev,
                'Target Return': target_return,
                'Portfolio Variance': portfolio_variance,
                'Portfolio Std Dev': portfolio_std_dev,
                **dict(zip(selected_assets, w.value))
            }, index=[0])
            mv_df = pd.concat([mv_df, new_row], ignore_index=True)
        else:
            # print(f"Problem status: {problem.status}")
            continue
            # for i, constraint in enumerate(constraints, 1):
            #     print(f"Constraint {i}: {str(constraint)}")

    return mv_df

def calculate_tracking_error(weights, asset_covar, asset_benchmark_covar, benchmark_var):
    portfolio_variance = np.dot(np.dot(weights, asset_covar), weights)
    cross_term = np.dot(weights, asset_benchmark_covar)
    tracking_error_variance = portfolio_variance - 2 * cross_term + benchmark_var
    if tracking_error_variance < 0:
        print(f"Tracking error variance is negative: {tracking_error_variance:.3f}")
    tracking_error = np.sqrt(tracking_error_variance)
    return tracking_error

def calculate_tracking_error_base(asset_returns, benchmark_returns, weights):
    portfolio_returns = np.dot(asset_returns, weights)
    return_differences = portfolio_returns - benchmark_returns
    tracking_error = np.std(return_differences) * np.sqrt(12)

    return tracking_error

def tracking_error_constraint(weights, asset_covar, asset_benchmark_covar, benchmark_var, tracking_error_limit):
    tracking_error = calculate_tracking_error(weights, asset_covar, asset_benchmark_covar, benchmark_var)
    return tracking_error_limit - tracking_error

def calculate_tracking_error_cvxpy(weights, asset_covar, asset_benchmark_covar, benchmark_var):
    portfolio_variance = cp.quad_form(weights, asset_covar)
    cross_term = weights.T @ asset_benchmark_covar
    tracking_error = cp.sqrt(portfolio_variance - 2 * cross_term + benchmark_var)
    return tracking_error

def tracking_error_constraint_cvxpy(weights, asset_covar, asset_benchmark_covar, benchmark_var, tracking_error_limit):
    tracking_error = calculate_tracking_error_cvxpy(weights, asset_covar, asset_benchmark_covar, benchmark_var)
    return tracking_error <= tracking_error_limit

def calculate_max_enb_frontier(mv_df, arithmetic_mu, selected_covar, selected_assets, x0, t_mt, bounds, country,
                               epsilon, lambda_coeff, rdf, benchmarks, tracking_error_limit, selected_date, additional_constraints,
                               tracking_error_increment=0.025, max_tracking_error=0.1, parameter_retry_flag=True, selected_benchmark='Flex'):
    calculator = EffectiveRankCalculator(algorithm=None,
                                         minRank=1,
                                         tickers=[],
                                         period=252 * 2,
                                         pctChangePeriod=21,
                                         covar_method='',
                                         resolution=None)

    n = len(arithmetic_mu)

    results_df = pd.DataFrame(columns=selected_assets)
    calculator.calculate(covarianceMatrix=selected_covar.to_numpy())

    combined_returns = rdf.copy()
    combined_returns[selected_benchmark] = benchmarks[selected_benchmark].pct_change().dropna()

    combined_returns = combined_returns.dropna()
    # combined_returns = combined_returns.loc[combined_returns.index <= selected_date]
    cov_matrix = combined_returns.cov() * 12
    asset_benchmark_covar = cov_matrix.loc[selected_assets, selected_benchmark]
    benchmark_var = np.var(combined_returns.loc[:, selected_benchmark]) * 12

    # start_index = len(target_stdevs) // 3
    start_index = 0

    target_stdevs_above = target_stdevs[start_index:]
    target_stdevs_below = target_stdevs[:start_index][::-1]

    penalty_coeff = lambda_coeff

    success_tracker = []

    erc_weights = optimization.get_erc_weights(selected_covar)

    while results_df.empty:
        if max_tracking_error is not None and tracking_error_limit > max_tracking_error:
            print(f"Reached maximum tracking error limit: {max_tracking_error}. No successful optimizations found.")
            break

        success_tracker.clear()

        initial_weights = np.repeat(1 / len(selected_assets), len(selected_assets))

        if len(target_stdevs_above) > 0:
            for target_vol in target_stdevs_above:

                # if target_vol >= 0.074 and target_vol <= 0.076:
                #     print('Break')

                mv_df['Difference'] = abs(mv_df['Portfolio Std Dev'] - target_vol)
                closest_row = mv_df.iloc[mv_df['Difference'].idxmin()]
                target_return = closest_row['Target Return']

                if math.isfinite(target_return):

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

                    # te_covar = calculate_tracking_error(weight_ref, cov_matrix.loc[selected_assets, selected_assets], asset_benchmark_covar, benchmark_var)
                    # te_base = calculate_tracking_error_base(combined_returns[selected_assets], combined_returns['40/60'], weight_ref)
                    # if round(te_covar, 3) != round(te_base, 3):
                    #    print(f"Tracking error mismatch: {te_covar:.3f} vs {te_base:.3f}")

                    constraints = [
                        {'type': 'eq', 'fun': constraint_sum_to_one},
                        {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                        {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
                    ]

                    if tracking_error_limit is not None:
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w: tracking_error_constraint(w, cov_matrix.loc[selected_assets, selected_assets], asset_benchmark_covar, benchmark_var, tracking_error_limit)
                        })

                    constraints = add_additional_constraints(constraints, additional_constraints,
                                                             library='scipy', country=country,
                                                             number_of_assets=n)
                    try:
                        # result = minimize(
                        #     enb_soft_constraint_objective,
                        #     weight_ref,
                        #     args=(cov_matrix.loc[selected_assets, selected_assets].to_numpy(), t_mt, weight_ref, "huber", penalty_coeff),
                        #     method='SLSQP',
                        #     bounds=bounds,
                        #     constraints=constraints,
                        #     options={'maxiter': 1000, 'ftol': 1e-10}
                        # )
                        norm = 'l1'
                        result = minimize(
                            weight_objective, x0=initial_weights, args=(erc_weights, norm), method='SLSQP',
                            constraints=constraints, bounds=bounds,
                            options={'maxiter': 10000, 'ftol': 1e-10}
                        )
                    except:
                        print('Error in optimization')

                    success_tracker.append(result.success)

                    if result.success:
                        maxenb_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
                        # initial_weights = maxenb_weights.to_numpy()
                        aligned_weights, aligned_mu = maxenb_weights.align(arithmetic_mu, join='inner')
                        portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

                        new_row = pd.DataFrame({'ENB': portfolio_properties['enb'], 'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0],
                                                'Sigma': portfolio_properties['sigma'], 'Tracking Error': calculate_tracking_error(result.x, cov_matrix.loc[selected_assets, selected_assets], asset_benchmark_covar, benchmark_var),
                                                'Diversification Ratio Squared': portfolio_properties['div_ratio_sqrd'], 'Effective Rank': calculator.effectiveRank,
                                                'Herfindahl Index': portfolio_properties['herfindahl_index'], 'Herfindahl Index RC': portfolio_properties['herfindahl_index_rc'],
                                                'ENB PCA': portfolio_properties['enb_pca'], 'Date': selected_date,
                                                **dict(zip(selected_assets, result.x))}, index=[0])
                        results_df = pd.concat([results_df, new_row], ignore_index=True)

        if results_df.empty:
            if parameter_retry_flag:
                tracking_error_limit += tracking_error_increment
                print(f"No successful optimization found. Increasing tracking error limit to {tracking_error_limit:.3f}")
            else:
                break

    formatted_date = selected_date.strftime('%Y-%m-%d')

    plt.figure(figsize=(10, 6))

    for i, target_vol in enumerate(target_stdevs_above):
        if success_tracker[i]:
            plt.scatter(target_vol, 1, color='green', label='Success' if i == 0 else "", marker='o', s=100, alpha=0.7)
        else:
            plt.scatter(target_vol, 0, color='red', label='Failure' if i == 0 else "", marker='x', s=100, alpha=0.7)

    plt.xlabel('Target Volatility')
    plt.ylabel('Optimization Status (1 = Success, 0 = Failure)')
    plt.title(f'Optimization Success for Target Volatilities - Date: {formatted_date}')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Failure', 'Success'])
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend()

    os.makedirs('./images/frontier_availability', exist_ok=True)
    plt.savefig(f'./images/frontier_availability/frontier_success_{formatted_date}.png')
    plt.close()

    # if len(target_stdevs_below) > 0:
    #     for target_vol in target_stdevs_below:
    #         mv_df['Difference'] = abs(mv_df['Portfolio Std Dev'] - target_vol)
    #         closest_row = mv_df.iloc[mv_df['Difference'].idxmin()]
    #         target_return = closest_row['Target Return']

    #         if math.isfinite(target_return):
    #             # new_row = pd.DataFrame({'Target Vol': target_vol, 'Target Return': target_return,
    #             #                      **dict(zip(selected_assets, closest_row[selected_assets]))}, index=[0])
    #             # mv_df = pd.concat([mv_df, new_row], ignore_index=True)

    #             ################################################################################################
    #             # Maximum ENB Portfolio
    #             def enb_soft_constraint_objective(w, *args):
    #                 # sigma, t_MT, weight_ref, norm, penalty_coeff = args
    #                 sigma, t_MT, weight_ref, norm, penalty_coeff = args
    #                 original_objective = EffectiveBets_scalar(w, sigma, t_MT)
    #                 penalty_term = penalty_coeff * weight_objective(w, weight_ref, norm)
    #                 return original_objective + penalty_term

    #             weight_ref = maxenb_weights.to_numpy()

    #             constraints = [
    #                 {'type': 'eq', 'fun': constraint_sum_to_one},
    #                 {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
    #                 {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
    #             ]

    #             if tracking_error_limit is not None:
    #                 constraints.append({
    #                     'type': 'ineq',
    #                     'fun': lambda w: tracking_error_constraint(w, selected_covar, asset_benchmark_covar, benchmark_var, tracking_error_limit)
    #                 })

    #             constraints = add_additional_constraints(constraints, additional_constraints,
    #                                                      library='scipy', country=country,
    #                                                      number_of_assets=n)

    #             result = minimize(
    #                 enb_soft_constraint_objective,
    #                 x0,
    #                 args=(selected_covar.to_numpy(), t_mt, weight_ref, "huber", penalty_coeff),
    #                 method='SLSQP',
    #                 bounds=bounds,
    #                 constraints=constraints,
    #                 options={'maxiter': 1000, 'ftol': 1e-10}
    #             )

    #             maxenb_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
    #             aligned_weights, aligned_mu = maxenb_weights.align(arithmetic_mu, join='inner')
    #             portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar)

    #             new_row = pd.DataFrame({'ENB': -result.fun ,'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0], 'Sigma': portfolio_properties['sigma'],
    #                                   **dict(zip(selected_assets, result.x))}, index=[0])
    #             results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df


def save_and_plot_results(results_df: pd.DataFrame, mv_df: pd.DataFrame, current_date: pd.Timestamp,
                          frontiers_dir: str, plots_dir: str, selected_assets: List[str], config, epsilon):
    mv_df = mv_df.sort_values(by='Target Vol')
    try:
        results_df = results_df.sort_values(by='Target Vol')
    except Exception as e:
        print(f"An error occurred while sorting results_df at date {current_date}: {e}")
    else:
        results_df.to_pickle(os.path.join(frontiers_dir, f"enb_{current_date.strftime('%Y%m%d')}"))
        mv_df['Adjusted Return'] = (1 - epsilon) * mv_df['Target Return']
        plot_charts(results_df, 'Max ENB Frontier', 'blue', f'enb_{current_date.date().strftime("%Y%m%d")}',
                    include_enb=True, selected_assets=selected_assets, selected_date=current_date, mv_df=mv_df,
                    plots_dir=plots_dir, config=config, epsilon=epsilon)

def process_date(args):
    selected_date, er, covar, rdf, benchmarks, frontiers_dir, plots_dir, config = args
    print(f"Processing date: {selected_date} for config: {config['Config']}")

    country = config['Country']
    epsilon = config['Epsilon']
    lambda_coeff = config['Lambda']
    additional_constraints = config['Additional Constraints']
    tracking_error_constraint = config['Tracking Error Constraint']
    tracking_error_limit = config['Tracking Error Limit']

    if additional_constraints == 'None':
        additional_constraints = None
    elif additional_constraints == 'Yes':
        additional_constraints = {'alternatives_upper_bound': 0.144,
                                  # 'gold_upper_bound': 0.1,
                                  'em_equities_upper_bound': 0.3,
                                  'hy_credit_upper_bound': 0.086,
                                  }
        # additional_constraints = {'alternatives_upper_bound': 0.20,
        #                           'em_equities_upper_bound': 0.3,
        #                           'hy_credit_upper_bound': 0.30,
        #                           'gold_upper_bound': 0.15}
    if tracking_error_constraint != 'Yes':
        tracking_error_limit = None

    selected_assets = get_selected_assets(er, selected_date)
    selected_assets = [asset for asset in selected_assets if asset != 'Cash']
    selected_covar = covar.loc[selected_assets, selected_assets]
    selected_er = er.loc[selected_date, selected_assets]

    stds = np.sqrt(np.diag(selected_covar))
    arithmetic_mu = selected_er + np.square(stds) / 2

    t_mt = torsion(selected_covar, 'minimum-torsion', method='exact')
    x0 = np.full((len(selected_assets),), 1 / len(selected_assets))

    bounds = [(0, 1) for _ in range(len(selected_assets))]

    mv_df = perform_mean_variance_optimization(arithmetic_mu, selected_covar, selected_assets, country,
                                               additional_constraints, tracking_error_limit, rdf,
                                               benchmarks)

    results_df = calculate_max_enb_frontier(mv_df, arithmetic_mu, selected_covar,
                                            selected_assets, x0, t_mt, bounds,
                                            country, epsilon, lambda_coeff, rdf, benchmarks, tracking_error_limit,
                                            selected_date, additional_constraints)

    save_and_plot_results(results_df, mv_df, selected_date, frontiers_dir, plots_dir, selected_assets, config, epsilon)

    plot_tracking_error_chart = False
    if plot_tracking_error_chart:
        benchmark_expected_return = arithmetic_mu['DM Equities'] * 0.4 + arithmetic_mu['Gov Bonds'] * 0.6
        # benchmark_expected_return = selected_er['DM Equities'] * 0.4665 + selected_er['Gov Bonds'] * 0.5335

        tracking_error_limits = np.arange(0.01, 0.11, 0.01)
        tracking_error_limits = np.insert(tracking_error_limits, 0, 0.00075)  # Add 0.05 at the beginning

        # target_volatility = 0.06274
        target_volatility = 0.08

        selected_rows = pd.DataFrame()

        for tracking_error_limit in tracking_error_limits:
            print(f"Running optimization with tracking error limit: {tracking_error_limit:.3f}")
            mv_df = perform_mean_variance_optimization(arithmetic_mu, selected_covar, selected_assets, country,
                                                       additional_constraints, tracking_error_limit, rdf,
                                                       benchmarks)

            results_df = calculate_max_enb_frontier(
                mv_df, arithmetic_mu, selected_covar, selected_assets, x0, t_mt, bounds,
                country, 0.5, lambda_coeff, rdf, benchmarks, tracking_error_limit,
                selected_date, additional_constraints, parameter_retry_flag=False, selected_benchmark='40/60'
            )

            if not results_df.empty:
                filtered_df = results_df[results_df['Sigma'] <= target_volatility]

                if not filtered_df.empty:
                    closest_row = filtered_df.loc[filtered_df['Sigma'].idxmax()]
                    selected_rows = selected_rows.append(closest_row, ignore_index=True)

        from matplotlib.ticker import PercentFormatter

        formatted_volatility = f"{target_volatility:.2%}"
        formatted_date = selected_date.strftime("%Y-%m-%d")

        plt.figure(figsize=(10, 6))
        plt.scatter(selected_rows['Tracking Error'], selected_rows['Return'], marker='o')

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fourth_color = default_colors[3]

        plt.scatter(0, benchmark_expected_return, color=fourth_color, marker='o', label='Benchmark Expected Return')

        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Tracking Error')
        plt.ylabel('Return')
        plt.title(f'Return vs. Tracking Error (Target Volatility = {formatted_volatility}, Selected Date = {formatted_date})')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        max_te = selected_rows['Tracking Error'].max()
        max_return = selected_rows['Return'].max()
        plt.xlim(right=max_te * 1.1)
        plt.ylim(top=max_return * 1.1)
        plt.grid(True)
        plt.show()

        plt.savefig(f'{plots_dir}/tracking_error_frontier_{selected_date.strftime("%Y%m%d")}_{config["Config"]}.png')

    return {"date": selected_date, "config": config['Config'], "data": "Results processed successfully"}

def run_configs(configs_to_run=None):
    all_configs = read_configs()

    if configs_to_run is None:
        configs_to_run = all_configs
    else:
        configs_to_run = [config for config in all_configs if config['Config'] in configs_to_run]

    num_cores = os.cpu_count()
    # num_cores = 1

    for config in configs_to_run:
        print(f"Running configuration: {config['Config']}")

        country = config['Country']
        frontiers_dir = f"./frontiers/config_{config['Config']}"
        plots_dir = f"./plots/config_{config['Config']}"
        os.makedirs(frontiers_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        if country == 'US' or country == 'UK' or country == 'JP':
            rdf = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="cov")
            rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
            rdf.set_index('Date', inplace=True)

            er = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="expected_gross_return")
            er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
            er.set_index('Date', inplace=True)
            er.loc[:"1973-01-31", "Gold"] = np.nan

            benchmark_returns = rdf.loc[:, ['Equities', 'Gov Bonds']]
            benchmark_returns.loc[:, '40/60'] = 0.4 * benchmark_returns['Equities'] + 0.6 * benchmark_returns['Gov Bonds']
            benchmark_returns.loc[:, '60/40'] = 0.6 * benchmark_returns['Equities'] + 0.4 * benchmark_returns['Gov Bonds']
            benchmark_returns.loc[:, 'Flex'] = 0.5335 * benchmark_returns['Equities'] + 0.4665 * benchmark_returns['Gov Bonds']

            benchmarks = (1 + benchmark_returns.loc[:, ['40/60', '60/40', 'Flex']]).cumprod()
        elif np.isnan(country):
            rdf = pd.read_excel(f"./data/2024-11-24 master_file.xlsx", sheet_name="cov")
            rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
            rdf = rdf.loc[rdf.loc[:, 'Date'] >= '1993-01-31', :]
            rdf.set_index('Date', inplace=True)

            er = pd.read_excel(f"./data/2024-11-24 master_file.xlsx", sheet_name="expected_gross_return")
            er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
            er.set_index('Date', inplace=True)
            er.loc[:"1973-01-31", "Gold"] = np.nan

            # benchmark_returns = read_benchmark_returns()

            benchmark_returns = rdf.loc[:, ['DM Equities', 'Gov Bonds']]
            benchmark_returns.loc[:, '40/60'] = 0.4 * benchmark_returns['DM Equities'] + 0.6 * benchmark_returns['Gov Bonds']
            benchmark_returns.loc[:, '60/40'] = 0.6 * benchmark_returns['DM Equities'] + 0.4 * benchmark_returns['Gov Bonds']
            benchmark_returns.loc[:, 'Flex'] = 0.5335 * benchmark_returns['DM Equities'] + 0.4665 * benchmark_returns['Gov Bonds']

            benchmarks = (1 + benchmark_returns.loc[:, ['40/60', '60/40', 'Flex']]).cumprod()

        const_covar = rdf.cov()
        covar = const_covar * 12

        # smaller_covar = covar.loc[['DM Equities', 'Gov Bonds'], ['DM Equities', 'Gov Bonds']]
        # weights = np.array([0.40, 0.60])
        # portfolio_variance = np.dot(weights.T, np.dot(smaller_covar, weights))
        # portfolio_volatility = np.sqrt(portfolio_variance)

        # start_date = pd.Timestamp('1900-01-31')
        # start_date = pd.Timestamp('2017-01-31')
        # start_date = pd.Timestamp('1998-10-31')
        start_date = pd.Timestamp('1875-01-31')
        # start_date = pd.Timestamp('2009-05-31')
        # start_date = pd.Timestamp('2024-10-31')
        dates = er.index.unique()[er.index.unique() >= start_date]

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            args_list = [(date, er, covar, rdf, benchmarks, frontiers_dir, plots_dir, config) for date in dates]
            results = list(executor.map(process_date, args_list))

        print(f"Completed configuration: {config['Config']}")
        print(results)

def read_configs(file_path='./Configs.xlsx'):
    configs = pd.read_excel(file_path, sheet_name='Sheet1')
    return configs.to_dict('records')

if __name__ == "__main__":
    run_configs(configs_to_run=configs_to_run)
