import numpy as np
import pandas as pd

from meucci.torsion import torsion

import matplotlib.pyplot as plt
import cvxpy as cp
from examples.effective_rank_calculator import EffectiveRankCalculator
import math
from meucci.EffectiveBets import EffectiveBets
import os
from scipy.optimize import minimize
from analysis.drawdowns import endpoint_mdd_lookup
import datetime
import re
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import sklearn
from portfolio_construction import calculations, optimization

target_stdevs = np.arange(0.01, 0.13, 0.0025)

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

def EffectiveBets_scalar(x, Sigma, t_MT):
    _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
    return -scalar_matrix.item()

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

def diversification_ratio_squared(w, sigma_port, standard_deviations):
    return (np.dot(w, standard_deviations) / sigma_port) ** 2

def pf_mu(weight, mu):
    return np.array([np.inner(weight, mu)])

def pf_sigma(weight, cov):
    weight = weight.ravel()
    return np.sqrt(np.einsum('i,ij,j->', weight, cov, weight))

def calculate_portfolio_properties(caaf_weights, arithmetic_mu, covar):
    aligned_weights, aligned_mu = caaf_weights.align(arithmetic_mu, join='inner')

    portfolio_arithmetic_mu = pf_mu(aligned_weights, aligned_mu)
    portfolio_sigma = pf_sigma(aligned_weights, covar)
    portfolio_geo_mu = portfolio_arithmetic_mu - 0.5 * portfolio_sigma ** 2
    portfolio_md = endpoint_mdd_lookup(portfolio_geo_mu, portfolio_sigma, frequency='M', percentile=5)

    div_ratio_squared = diversification_ratio_squared(aligned_weights, portfolio_sigma, np.sqrt(np.diag(covar)))

    covar = np.asarray(covar)

    t_mt = torsion(covar, 'minimum-torsion', method='exact')
    p, enb = EffectiveBets(aligned_weights.to_numpy(), covar, t_mt)

    t_pca = torsion(covar, 'pca', method='exact')
    p_pca, enb_pca = EffectiveBets(aligned_weights.to_numpy(), covar, t_pca)

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

def perform_mean_variance_optimization(arithmetic_mu, selected_covar, selected_assets,
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
    if additional_constraints:
        constraints.append(w[5] <= additional_constraints['alternatives_upper_bound'])
        constraints.append((w[0] + w[1]) * additional_constraints['em_equities_upper_bound'] >= w[1])
        constraints.append(w[2] <= additional_constraints['hy_credit_upper_bound'])

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
        if additional_constraints:
            constraints.append(w[5] <= additional_constraints['alternatives_upper_bound'])
            constraints.append((w[0] + w[1]) * additional_constraints['em_equities_upper_bound'] >= w[1])
            constraints.append(w[2] <= additional_constraints['hy_credit_upper_bound'])

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


def calculate_max_enb_frontier(mv_df, arithmetic_mu, selected_covar, selected_assets, x0, t_mt, bounds,
                               epsilon, lambda_coeff, rdf, benchmarks, tracking_error_limit, selected_date, additional_constraints,
                               tracking_error_increment=0.025, max_tracking_error=None):
    calculator = EffectiveRankCalculator(algorithm=None,
                                         minRank=1,
                                         tickers=[],
                                         period=252 * 2,
                                         pctChangePeriod=21,
                                         covar_method='',
                                         resolution=None)

    n = len(arithmetic_mu)

    maxenb_weights = pd.Series(x0)
    results_df = pd.DataFrame(columns=selected_assets)
    benchmark_returns = benchmarks['Flex'].pct_change().dropna()

    calculator.calculate(covarianceMatrix=selected_covar)

    combined_returns = rdf.copy()
    combined_returns['Flex'] = benchmarks['Flex'].pct_change().dropna()

    combined_returns = combined_returns.dropna()
    # combined_returns = combined_returns.loc[combined_returns.index <= selected_date]
    cov_matrix = combined_returns.cov() * 12
    asset_benchmark_covar = cov_matrix.loc[selected_assets, 'Flex']
    benchmark_var = np.var(combined_returns.loc[:, 'Flex']) * 12

    # start_index = len(target_stdevs) // 3
    start_index = 0

    target_stdevs_above = target_stdevs[start_index:]
    target_stdevs_below = target_stdevs[:start_index][::-1]

    penalty_coeff = lambda_coeff

    success_tracker = []

    current_tracking_error_limit = tracking_error_limit

    while results_df.empty:
        if max_tracking_error is not None and current_tracking_error_limit > max_tracking_error:
            print(f"Reached maximum tracking error limit: {max_tracking_error}. No successful optimizations found.")
            break

        success_tracker.clear()

        if len(target_stdevs_above) > 0:
            for target_vol in target_stdevs_above:
                mv_df['Difference'] = abs(mv_df['Portfolio Std Dev'] - target_vol)
                closest_row = mv_df.iloc[mv_df['Difference'].idxmin()]
                target_return = closest_row['Target Return']

                if math.isfinite(target_return):

                    def enb_soft_constraint_objective(w, *args):
                        sigma, t_MT, weight_ref, norm, penalty_coeff = args
                        original_objective = EffectiveBets_scalar(w, sigma, t_MT)
                        penalty_term = penalty_coeff * weight_objective(w, weight_ref, norm)
                        return original_objective + penalty_term

                    weight_ref = maxenb_weights.to_numpy()

                    # te_covar = calculate_tracking_error(weight_ref, cov_matrix.loc[selected_assets, selected_assets], asset_benchmark_covar, benchmark_var)
                    # te_base = calculate_tracking_error_base(combined_returns[selected_assets], combined_returns['40/60'], weight_ref)
                    # if round(te_covar, 3) != round(te_base, 3):
                    #    print(f"Tracking error mismatch: {te_covar:.3f} vs {te_base:.3f}")

                    constraints = [
                        {'type': 'eq', 'fun': constraint_sum_to_one},
                        {'type': 'eq', 'fun': lambda w: volatility_constraint(w, selected_covar, target_vol)},
                        {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return}
                    ]

                    if current_tracking_error_limit is not None:
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w: tracking_error_constraint(w, cov_matrix.loc[selected_assets, selected_assets], asset_benchmark_covar, benchmark_var, current_tracking_error_limit)
                        })

                    if additional_constraints:
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w: additional_constraints['alternatives_upper_bound'] - w[5],
                            'name': 'Alternatives Constraint'
                        })
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w: (w[0] + w[1]) * additional_constraints['em_equities_upper_bound'] - w[1],
                            'name': 'EM Equities Constraint'
                        })
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w: additional_constraints['hy_credit_upper_bound'] - w[2],
                            'name': 'HY Credit Constraint'
                        })

                    try:
                        result = minimize(
                            enb_soft_constraint_objective,
                            weight_ref,
                            args=(cov_matrix.loc[selected_assets, selected_assets].to_numpy(), t_mt, weight_ref, "huber", penalty_coeff),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-10}
                        )
                    except:
                        print('Error in optimization')

                    success_tracker.append(result.success)

                    if result.success:
                        maxenb_weights = pd.Series({arithmetic_mu.index[i]: result.x[i] for i in range(n)})
                        aligned_weights, aligned_mu = maxenb_weights.align(arithmetic_mu, join='inner')
                        portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, selected_covar.to_numpy())

                        new_row = pd.DataFrame({'ENB': -result.fun, 'Target Vol': target_vol, 'Return': portfolio_properties['arithmetic_mu'][0],
                                                'Sigma': portfolio_properties['sigma'], 'Tracking Error': calculate_tracking_error(result.x, cov_matrix.loc[selected_assets, selected_assets], asset_benchmark_covar, benchmark_var),
                                                'Diversification Ratio Squared': portfolio_properties['div_ratio_sqrd'], 'Effective Rank': calculator.effectiveRank,
                                                'Herfindahl Index': portfolio_properties['herfindahl_index'], 'Herfindahl Index RC': portfolio_properties['herfindahl_index_rc'],
                                                'ENB PCA': portfolio_properties['enb_pca'], 'Date': selected_date,
                                                **dict(zip(selected_assets, result.x))}, index=[0])
                        results_df = pd.concat([results_df, new_row], ignore_index=True)

        if results_df.empty:
            epsilon += tracking_error_increment
            print(f"No successful optimization found. Increasing epsilon to {epsilon:.3f}")

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

def calc_two_stage_weights(
        returns, exp_rets, target_md=None, target_volatility=None, epsilon=0, erc_weights=None, norm="huber", weight_bounds=(0.0, 1.0),
        additional_constraints=None, covar_method="standard", periodicity=12, const_covar=None, initial_weights_two_stage=None, options=None,
):
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

    def weight_objective(weight, weight_ref, norm, delta=0.1):
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

    if target_md is not None and target_volatility is not None:
        raise ValueError("Both target_md and target_volatility cannot be set.")

    exp_rets.dropna(inplace=True)
    n = len(exp_rets)

    # calc covariance matrix
    if covar_method == "ledoit-wolf":
        covar = sklearn.covariance.ledoit_wolf(returns)[0]
    elif covar_method == "standard":
        covar = returns.cov()
    elif covar_method == "constant":
        if const_covar is None:
            raise ValueError("const_covar must be provided if covar_method is constant")
        covar = const_covar.loc[list(exp_rets.index), list(exp_rets.index)]
    else:
        raise NotImplementedError("covar_method not implemented")

    covar *= periodicity

    if covar_method == "constant":
        stds = np.sqrt(np.diag(covar))
    else:
        stds = returns.std() * np.sqrt(periodicity)
    arithmetic_mu = exp_rets + np.square(stds) / 2

    initial_weights = np.ones([n]) / n

    if initial_weights_two_stage is None:
        initial_weights_two_stage = np.ones([n]) / n

    bounds = [weight_bounds for i in range(n)]

    constraints = [
        {"type": "eq", "fun": lambda W: sum(W) - 1.0},  # sum of weights must be equal to 1
        {"type": "eq", "fun": lambda W: volatility_constraint(W, covar, target_volatility)}  # volatility constraint
    ]
    if additional_constraints:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: additional_constraints['alternatives_upper_bound'] - w[5],
            'name': 'Alternatives Constraint'
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: (w[0] + w[1]) * additional_constraints['em_equities_upper_bound'] - w[1],
            'name': 'EM Equities Constraint'
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: additional_constraints['hy_credit_upper_bound'] - w[2],
            'name': 'HY Credit Constraint'
        })

    mv_return_optimum = minimize(
        return_objective,
        initial_weights,
        (-arithmetic_mu,),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )
    # check if success
    if not mv_return_optimum.success:
        raise Exception(mv_return_optimum.message)

    target_return = -mv_return_optimum.fun

    constraints = [
        {"type": "eq", "fun": lambda w: sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: return_objective(w, arithmetic_mu) - (1 - epsilon) * target_return},
        {'type': 'eq', 'fun': lambda w: volatility_constraint(w, covar, target_volatility)}
    ]
    if additional_constraints:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: additional_constraints['alternatives_upper_bound'] - w[5],
            'name': 'Alternatives Constraint'
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: (w[0] + w[1]) * additional_constraints['em_equities_upper_bound'] - w[1],
            'name': 'EM Equities Constraint'
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: additional_constraints['hy_credit_upper_bound'] - w[2],
            'name': 'HY Credit Constraint'
        })

    # weight_ref = np.array(erc_weights[list(exp_rets.index)])
    weight_ref = erc_weights

    optimum = minimize(
        weight_objective, x0=initial_weights_two_stage, args=(weight_ref, norm), method='SLSQP',
        constraints=constraints, bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    twostage_weights = pd.Series({exp_rets.index[i]: optimum.x[i] for i in range(n)})
    aligned_weights, aligned_mu = twostage_weights.align(arithmetic_mu, join='inner')
    portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, covar)

    return twostage_weights, portfolio_properties

def percentage_formatter(x, pos):
    """Format y-axis values as percentages."""
    return f"{100 * x:.0f}%"

def rgb_to_tuple(rgb_str):
    """Converts 'rgb(a, b, c)' string to a tuple of floats scaled to [0, 1]"""
    nums = re.findall(r'\d+', rgb_str)
    return tuple(int(num) / 255.0 for num in nums)

color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "DM Equities": "rgb(64, 75, 151)",
    "EM Equities": "rgb(131, 140, 202)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Alternatives": "rgb(160, 84, 66)",
    "Gold": "rgb(216, 169, 23)"
    }

# Convert the colors in color_mapping to tuples
tuple_color_mapping = {key: rgb_to_tuple(value) for key, value in color_mapping.items()}

epsilon = 0.1
lambda_coeff = 40
additional_constraints = {'alternatives_upper_bound': 0.5,
                          'em_equities_upper_bound': 0.3,
                          'hy_credit_upper_bound': 0.086, }
tracking_error_limit = None

selected_assets =  ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']

selected_covar = np.array([[ 2.36364756e-02,  2.11099011e-02,  6.81061330e-03,
         1.41877493e-04,  1.07958144e-03,  7.18911263e-04],
       [ 2.11099011e-02,  4.61147288e-02,  7.62190506e-03,
        -7.30186672e-04,  4.43574686e-03,  1.37810516e-03],
       [ 6.81061330e-03,  7.62190506e-03,  7.37626267e-03,
         8.52843164e-04,  5.41988480e-05,  2.99100916e-04],
       [ 1.41877493e-04, -7.30186672e-04,  8.52843164e-04,
         1.99695438e-03,  1.36140290e-04, -2.88818140e-05],
       [ 1.07958144e-03,  4.43574686e-03,  5.41988480e-05,
         1.36140290e-04,  3.23385708e-02,  8.13024785e-04],
       [ 7.18911263e-04,  1.37810516e-03,  2.99100916e-04,
        -2.88818140e-05,  8.13024785e-04,  3.10299539e-03]])
selected_covar = pd.DataFrame(selected_covar)
selected_covar.columns = selected_assets
selected_covar.index = selected_assets

selected_assets = selected_covar.columns
std_devs = np.sqrt(np.diag(selected_covar))
corr_matrix = selected_covar / np.outer(std_devs, std_devs)
corr_matrix = pd.DataFrame(corr_matrix, index=selected_assets, columns=selected_assets)

# std_devs[5] = 0.15

new_covar = corr_matrix * np.outer(std_devs, std_devs)
new_covar = pd.DataFrame(new_covar, index=selected_assets, columns=selected_assets)
selected_covar = new_covar

selected_er = np.array([0.04927393, 0.06791977, 0.03650494, 0.01930767, 0.01496375,
       0.05514604])
stds = np.sqrt(np.diag(selected_covar))
arithmetic_mu = selected_er + np.square(stds) / 2

arithmetic_mu = pd.Series(arithmetic_mu, index=selected_assets)

t_mt = torsion(selected_covar, 'minimum-torsion', method='exact')
x0 = np.full((len(selected_assets),), 1 / len(selected_assets))

bounds = [(0, 1) for _ in range(len(selected_assets))]

selected_date = datetime.date(2024, 11, 12)

rdf = pd.read_excel(f"./data/2024-08-31 master_file.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf = rdf.loc[rdf.loc[:, 'Date'] >= '1993-01-31', :]
rdf.set_index('Date', inplace=True)

er = pd.read_excel(f"./data/2024-08-31 master_file.xlsx", sheet_name="expected_gross_return")
er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
er.set_index('Date', inplace=True)
er.loc[:"1973-01-31", "Gold"] = np.nan

benchmark_returns = rdf.loc[:, ['DM Equities', 'Gov Bonds']]
benchmark_returns.loc[:, '40/60'] = 0.4 * benchmark_returns['DM Equities'] + 0.6 * benchmark_returns['Gov Bonds']
benchmark_returns.loc[:, '60/40'] = 0.6 * benchmark_returns['DM Equities'] + 0.4 * benchmark_returns['Gov Bonds']
benchmark_returns.loc[:, 'Flex'] = 0.5335 * benchmark_returns['DM Equities'] + 0.4665 * benchmark_returns['Gov Bonds']

benchmarks = (1 + benchmark_returns.loc[:, ['40/60', '60/40', 'Flex']]).cumprod()


mv_df = perform_mean_variance_optimization(arithmetic_mu, selected_covar, selected_assets,
                                           additional_constraints, tracking_error_limit, rdf,
                                           benchmarks)

results_df = calculate_max_enb_frontier(mv_df, arithmetic_mu, selected_covar,
                                        selected_assets, x0, t_mt, bounds,
                                        epsilon, lambda_coeff, rdf, benchmarks, tracking_error_limit,
                                        selected_date, additional_constraints)
results_df.set_index('Sigma', inplace=True)

columns_to_plot = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
weights_data = results_df[columns_to_plot].fillna(0).astype(float)
sigma = results_df.index

def rgb_to_hex(rgb_str):
    rgb = tuple(map(int, rgb_str[4:-1].split(', ')))
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

colors = [rgb_to_hex(color_mapping[col]) for col in columns_to_plot]

plt.figure(figsize=(10, 6))
plt.stackplot(sigma, weights_data.T, labels=columns_to_plot, colors=colors)
plt.xlabel("Sigma")
plt.ylabel("Weights")
plt.title("Portfolio Weights vs Sigma")
plt.legend(loc="upper left")
plt.show()

erc_weights = optimization.get_erc_weights(selected_covar)

# two_stage_weights, _ = calc_two_stage_weights(rdf, arithmetic_mu, target_volatility=0.07, epsilon=0.1, erc_weights=erc_weights, norm="l1", additional_constraints=additional_constraints, covar_method="standard", periodicity=12, const_covar=selected_covar, initial_weights_two_stage=x0)

# Define the range of Sigma values
sigma_values = np.arange(0.0375, 0.13, 0.0025)  # Adjust the range and step as needed

# Initialize an empty list to store the results
results = []

# Loop over each Sigma (target volatility) value
for sigma in sigma_values:
    # Calculate the two-stage weights for each Sigma
    weights, _ = calc_two_stage_weights(
        rdf,
        arithmetic_mu,
        target_volatility=sigma,
        epsilon=0.1,
        erc_weights=erc_weights,
        norm="l1",
        additional_constraints=additional_constraints,
        covar_method="standard",
        periodicity=12,
        const_covar=selected_covar,
        initial_weights_two_stage=x0
    )

    # Add Sigma (target volatility) as well as Date
    weights["Sigma"] = sigma
    weights["Date"] = selected_date

    # Append the weights to the results list
    results.append(weights)

# Convert results list to a DataFrame
results_df = pd.DataFrame(results)

# Set Sigma as the index
results_df.set_index("Sigma", inplace=True)

print(results_df)

columns_to_plot = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
weights_data = results_df[columns_to_plot].fillna(0).astype(float)
sigma = results_df.index

plt.figure(figsize=(10, 6))
plt.stackplot(sigma, weights_data.T, labels=columns_to_plot, colors=colors)
plt.xlabel("Sigma")
plt.ylabel("Weights")
plt.title("Portfolio Weights vs Sigma")
plt.legend(loc="upper left")
plt.show()

# def calc_two_stage_weights(
#         returns, exp_rets, target_md=None, target_volatility=None, epsilon=0, erc_weights=None, norm="huber", weight_bounds=(0.0, 1.0),
#         additional_constraints=None, covar_method="standard", periodicity=12, const_covar=None, initial_weights_two_stage=None, options=None,
# )

# Function to calculate risk contributions given weights and covariance matrix
def calculate_risk_contribution(weights, cov_matrix):
    # Convert weights to numpy array
    weights = np.array(weights)
    # Portfolio variance
    portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights.T)
    # Marginal contributions to risk
    marginal_contribs = cov_matrix @ weights / portfolio_volatility
    # Risk contributions
    risk_contributions = weights * marginal_contribs
    return risk_contributions


# Initialize an empty list to store risk contributions
risk_contrib_results = []

# Loop over each Sigma (target volatility) level in results_df
for sigma, row in results_df.iterrows():
    # Extract weights for the assets at this Sigma level
    weights = row[columns_to_plot]  # Adjust `columns_to_plot` as per your asset columns
    # Calculate risk contributions for the given weights and covariance matrix
    risk_contributions = calculate_risk_contribution(weights, selected_covar)

    # Create a dictionary of results with Sigma as the index and asset risk contributions as values
    risk_contrib_dict = dict(zip(columns_to_plot, risk_contributions))
    risk_contrib_dict["Sigma"] = sigma
    # Append to the list
    risk_contrib_results.append(risk_contrib_dict)

# Convert risk contributions list to DataFrame
risk_contrib_df = pd.DataFrame(risk_contrib_results).set_index("Sigma")

# Display the risk contributions DataFrame
print(risk_contrib_df)

columns_to_plot = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
weights_data = results_df[columns_to_plot].fillna(0).astype(float)
sigma = results_df.index

plt.figure(figsize=(10, 6))
plt.stackplot(sigma, risk_contrib_df.T, labels=columns_to_plot, colors=colors)
plt.xlabel("Sigma")
plt.ylabel("Risk Contributions")
plt.title("Sigma vs Risk Contributions")
plt.legend(loc="upper left")
plt.show()
