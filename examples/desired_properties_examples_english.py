from meucci.EffectiveBets import EffectiveBets
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.drawdowns import endpoint_mdd_lookup
from meucci.torsion import torsion
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib

from portfolio_construction import calculations, optimization

from reporting.tools.style import set_clbrm_style
set_clbrm_style(caaf_colors=True)

matplotlib.rcParams.update({'font.size': 14})

# Convert RGB strings to hexadecimal
def rgb_to_hex(rgb_str):
    rgb = rgb_str.replace('rgb', '').replace('(', '').replace(')', '').split(',')
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

default_colors = [
    (64 / 255, 75 / 255, 151 / 255),
    (154 / 255, 183 / 255, 235 / 255),
    (144 / 255, 143 / 255, 74 / 255),
    (216 / 255, 169 / 255, 23 / 255),
    (160 / 255, 84 / 255, 66 / 255),
    (189 / 255, 181 / 255, 19 / 255),
    (144 / 255, 121 / 255, 65 / 255)
]

color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Alternatives": "rgb(160, 84, 66)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)"
}

hex_colors = {k: rgb_to_hex(v) for k, v in color_mapping.items()}

def diversification_ratio_squared(w, sigma_port, standard_deviations):
    return (np.dot(w, standard_deviations) / sigma_port) ** 2

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

def soft_constraint_objective(w, *args):
    # sigma, t_MT, weight_ref, norm, penalty_coeff = args
    sigma, t_MT, weight_ref, norm, penalty_coeff = args
    original_objective = EffectiveBets_scalar(w, sigma, t_MT)
    penalty_term = penalty_coeff * weight_objective(w, weight_ref, norm)
    return original_objective + penalty_term

def constraint_sum_to_one(x):
    return np.sum(x) - 1
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
    p, enb = EffectiveBets(aligned_weights.to_numpy(), covar, t_mt)

    # Compile portfolio properties into a pandas Series
    portfolio_properties = pd.Series({
        'arithmetic_mu': portfolio_arithmetic_mu,
        'sigma': portfolio_sigma,
        'md': portfolio_md,
        'enb': enb[0, 0],
        'div_ratio_sqrd': div_ratio_squared
    })

    return portfolio_properties

def to_percent(y, position):
    # Multiply the tick value by 100 and format as a string with a percentage sign
    return "{:.0f}%".format(100 * y)

target_vol = 0.07

# Asset 1, Asset 2, Equities, Credit, Government Bonds, Gold
asset_names = ["Asset Class 1", "Asset Class 2", "Equities", "High Yield", "Government Bonds", "Gold"]
color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "Government Bonds": "rgb(144, 143, 74)",
    "Asset Class 1" : "rgb(160, 84, 66)",
    "Asset Class 2": "rgb(180, 84, 66)",
    "High Yield": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)"
}

extended_arithmetic_mu = np.array([0.06, 0.061, 0.06, 0.04, 0.02, 0.02])
expected_returns = pd.Series(data=extended_arithmetic_mu,
                             index=asset_names)

extended_volatilities = np.array([0.1, 0.1, 0.15, 0.09, 0.05, 0.18])
correlation_matrix = np.array([[ 1.  ,  0.99 ,  0.12,  0.15,  0.2 ,  0.1 ],
                               [ 0.99 ,  1.  ,  0.12,  0.15,  0.2 ,  0.1 ],
                               [ 0.12,  0.12,  1.  ,  0.52 ,  0.02, 0.04],
                               [ 0.15,  0.15,  0.52,  1.  ,  0.22,  0.00],
                               [ 0.2 ,  0.2 ,  0.02 , 0.22 ,  1.  , 0.02],
                               [ 0.1 ,  0.1 ,  0.04 , 0.00 ,  0.02 ,  1.  ]])
volatility_outer_product = np.outer(extended_volatilities, extended_volatilities)
extended_sigma = correlation_matrix * volatility_outer_product
covariance_matrix = pd.DataFrame(data=extended_sigma,
                                 columns=asset_names,
                                 index=asset_names)

# Updating the number of assets and bounds
extended_n = len(extended_arithmetic_mu)
extended_bounds = [(0, 1) for _ in range(extended_n)]


def calculate_minimum_variance_portfolio(cov_matrix):
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    weights = np.dot(inv_cov_matrix, ones) / np.dot(ones.T, np.dot(inv_cov_matrix, ones))
    return weights

# Calculate the Minimum Variance Portfolio weights
min_var_weights_analytical = calculate_minimum_variance_portfolio(extended_sigma)

equal_weights = np.repeat(1 / extended_n, extended_n)

weight_ref = min_var_weights_analytical
penalty_coeff = 5

# sigma = np.array([[0.0225 , 0.02025],
#                   [0.02025, 0.0225 ]])
# arithmetic_mu = np.array([0.08 , 0.082])
# bounds = [(0, 1), (0, 1)]

epsilon = 0.01
n = 2

# Compute eigenvalues
eigenvalues = np.linalg.eigvalsh(extended_sigma[2:, 2:])

# Check if the matrix is positive semidefinite
is_positive_semidefinite = np.all(eigenvalues >= 0)

################################################################################
# Target Return Calculation

constraints = (
           {'type': 'eq', 'fun': constraint_sum_to_one},
           {'type': 'eq', 'fun': lambda w: volatility_constraint(w, extended_sigma, target_vol)}
)

result = minimize(
       lambda w: -pf_mu(w, extended_arithmetic_mu),  # We minimize the negative return to maximize the return
       equal_weights,
       method='SLSQP',
       bounds=extended_bounds,
       constraints=constraints,
       options={'maxiter': 10_000, 'ftol': 1e-15}
)

print(result.x)
target_return = -result.fun
mv_weights = pd.Series({asset_names[i]: result.x[i] for i in range(extended_n)})

constraints = (
    {'type': 'eq', 'fun': constraint_sum_to_one},
    {'type': 'eq', 'fun': lambda w: volatility_constraint(w, extended_sigma, target_vol)},
    {'type': 'ineq', 'fun': lambda w: return_objective(w, extended_arithmetic_mu) - (1 - epsilon) * target_return}
)

t_MT = torsion(extended_sigma, 'minimum-torsion')

result = minimize(
    soft_constraint_objective,
    min_var_weights_analytical,
    args=(extended_sigma, t_MT, weight_ref, "huber", penalty_coeff),
    method='SLSQP',
    bounds=extended_bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-10}
)

maxenb_weights = pd.Series({asset_names[i]: result.x[i] for i in range(extended_n)})
extended_arithmetic_mu = pd.Series({asset_names[i]: extended_arithmetic_mu[i] for i in range(extended_n)})
aligned_weights, aligned_mu = maxenb_weights, extended_arithmetic_mu
portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, extended_sigma)

# Convert the color mapping from 'rgb' to hexadecimal for compatibility with matplotlib
color_mapping_hex = {asset: '#' + ''.join(f'{int(c):02x}' for c in color.strip('rgb()').split(', ')) for asset, color in color_mapping.items()}

# Prepare DataFrame for seaborn plot
df = pd.DataFrame({'Asset Class': maxenb_weights.index, 'Weight': maxenb_weights.values})

matplotlib.rcParams.update({'font.size': 14})  # Adjust this value as needed
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('Maximum ENB Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)
plt.show()

# Prepare DataFrame for seaborn plot
df = pd.DataFrame({'Asset Class': mv_weights.index, 'Weight': mv_weights.values})

matplotlib.rcParams.update({'font.size': 14})  # Adjust this value as needed
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('Mean-Variance Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)
plt.show()

erc_weights = optimization.get_erc_weights(extended_sigma)
df = pd.DataFrame({'Asset Class': asset_names, 'Weight': erc_weights})

plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('ERC Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)
plt.show()


erc_weights = optimization.get_erc_weights(extended_sigma)
target_md = 0.39
mv_frontier = optimization.get_mv_frontier(mu=extended_arithmetic_mu, cov=extended_sigma, query_points=1000, target_md=target_md)

mv_weights = mv_frontier['Optimal Portfolio Weights']
caam_frontier = erc_weights * 0.5 + mv_frontier['Portfolio Weights'] * 0.5
caam_weights = erc_weights * 0.5 + mv_weights * 0.5
optimal_moments = calculations.pf_moments(weight=caam_weights, mu=extended_arithmetic_mu, is_geo=False, cov=extended_sigma)

portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, extended_sigma)

df = pd.DataFrame({'Asset Class': asset_names, 'Weight': caam_weights})

plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('CAAF 1.0 Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)
plt.show()


####################################################################################################
####################################################################################################
# Low Expected Return Example

target_vol = 0.07

# Asset 1, Asset 2, Equities, Credit, Government Bonds, Gold
asset_names = ["Asset Class 1", "Equities", "High Yield", "Government Bonds", "Gold"]
color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "Government Bonds": "rgb(144, 143, 74)",
    "Asset Class 1": "rgb(160, 84, 66)",
    "High Yield": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)"
}

extended_arithmetic_mu = np.array([-0.15, 0.06, 0.04, 0.02, 0.02])
extended_volatilities = np.array([0.1, 0.15, 0.09, 0.05, 0.18])

correlation_matrix = np.array([[1.  ,  0.12,  0.15,  0.2 ,  0.1 ],
                               [0.12,  1.  ,  0.52 ,  0.02, 0.04],
                               [0.15,  0.52,  1.  ,  0.22,  0.00],
                               [0.2 ,  0.02 , 0.22 ,  1.  , 0.02],
                               [0.1 ,  0.04 , 0.00 ,  0.02 ,  1.  ]])

volatility_outer_product = np.outer(extended_volatilities, extended_volatilities)
extended_sigma = correlation_matrix * volatility_outer_product


# Updating the number of assets and bounds
extended_n = len(extended_arithmetic_mu)
extended_bounds = [(0, 1) for _ in range(extended_n)]


def calculate_minimum_variance_portfolio(cov_matrix):
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones = np.ones(len(cov_matrix))
    weights = np.dot(inv_cov_matrix, ones) / np.dot(ones.T, np.dot(inv_cov_matrix, ones))
    return weights

# Calculate the Minimum Variance Portfolio weights
min_var_weights_analytical = calculate_minimum_variance_portfolio(extended_sigma)

equal_weights = np.repeat(1 / extended_n, extended_n)

weight_ref = min_var_weights_analytical
penalty_coeff = 5

# sigma = np.array([[0.0225 , 0.02025],
#                   [0.02025, 0.0225 ]])
# arithmetic_mu = np.array([0.08 , 0.082])
# bounds = [(0, 1), (0, 1)]

epsilon = 0.01
n = 2

# Compute eigenvalues
eigenvalues = np.linalg.eigvalsh(extended_sigma[2:, 2:])

# Check if the matrix is positive semidefinite
is_positive_semidefinite = np.all(eigenvalues >= 0)

################################################################################
# Target Return Calculation

constraints = (
           {'type': 'eq', 'fun': constraint_sum_to_one},
           {'type': 'eq', 'fun': lambda w: volatility_constraint(w, extended_sigma, target_vol)}
)

result = minimize(
       lambda w: -pf_mu(w, extended_arithmetic_mu),  # We minimize the negative return to maximize the return
       equal_weights,
       method='SLSQP',
       bounds=extended_bounds,
       constraints=constraints,
       options={'maxiter': 10_000, 'ftol': 1e-15}
)

print(result.x)
target_return = -result.fun
mv_weights = pd.Series({asset_names[i]: result.x[i] for i in range(extended_n)})

constraints = (
    {'type': 'eq', 'fun': constraint_sum_to_one},
    {'type': 'eq', 'fun': lambda w: volatility_constraint(w, extended_sigma, target_vol)},
    {'type': 'ineq', 'fun': lambda w: return_objective(w, extended_arithmetic_mu) - (1 - epsilon) * target_return}
)

t_MT = torsion(extended_sigma, 'minimum-torsion')

result = minimize(
    soft_constraint_objective,
    min_var_weights_analytical,
    args=(extended_sigma, t_MT, weight_ref, "huber", penalty_coeff),
    method='SLSQP',
    bounds=extended_bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-10}
)

maxenb_weights = pd.Series({asset_names[i]: result.x[i] for i in range(extended_n)})
extended_arithmetic_mu = pd.Series({asset_names[i]: extended_arithmetic_mu[i] for i in range(extended_n)})
aligned_weights, aligned_mu = maxenb_weights, extended_arithmetic_mu
portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, extended_sigma)

# Convert the color mapping from 'rgb' to hexadecimal for compatibility with matplotlib
color_mapping_hex = {asset: '#' + ''.join(f'{int(c):02x}' for c in color.strip('rgb()').split(', ')) for asset, color in color_mapping.items()}

# Prepare DataFrame for seaborn plot
df = pd.DataFrame({'Asset Class': maxenb_weights.index, 'Weight': maxenb_weights.values})

matplotlib.rcParams.update({'font.size': 14})  # Adjust this value as needed
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('Maximum ENB Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)
plt.show()

erc_weights = optimization.get_erc_weights(extended_sigma)
target_md = 0.39
mv_frontier = optimization.get_mv_frontier(mu=extended_arithmetic_mu, cov=extended_sigma, query_points=1000, target_md=target_md)

mv_weights = mv_frontier['Optimal Portfolio Weights']

# Prepare DataFrame for seaborn plot
df = pd.DataFrame({'Asset Class': asset_names, 'Weight': mv_weights})

matplotlib.rcParams.update({'font.size': 14})  # Adjust this value as needed
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('Mean-Variance Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)
plt.show()


erc_weights = optimization.get_erc_weights(extended_sigma)
df = pd.DataFrame({'Asset Class': asset_names, 'Weight': erc_weights})

plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('ERC Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)
plt.show()



erc_weights = optimization.get_erc_weights(extended_sigma)
target_md = 0.39
mv_frontier = optimization.get_mv_frontier(mu=extended_arithmetic_mu, cov=extended_sigma, query_points=1000, target_md=target_md)

mv_weights = mv_frontier['Optimal Portfolio Weights']
caam_frontier = erc_weights * 0.5 + mv_frontier['Portfolio Weights'] * 0.5
caam_weights = erc_weights * 0.5 + mv_weights * 0.5
optimal_moments = calculations.pf_moments(weight=caam_weights, mu=extended_arithmetic_mu, is_geo=False, cov=extended_sigma)

portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, extended_sigma)

df = pd.DataFrame({'Asset Class': asset_names, 'Weight': caam_weights})

plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('CAAF 1.0 Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.ylim(0, 0.55)

plt.show()


print("Break")