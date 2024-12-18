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
import sklearn

from portfolio_classes import calculations, optimization, CAAF

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
    "Equity": "rgb(64, 75, 151)",
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
asset_names = ["Asset Class 1", "Asset Class 2", "Equity", "HY Credit", "Gov Bonds", "Gold"]
color_mapping = {
    "Equity": "rgb(64, 75, 151)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Asset Class 1": "rgb(160, 84, 66)",
    "Asset Class 2": "rgb(180, 84, 66)",
    "HY Credit": "rgb(154, 183, 235)",
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
plt.title('CAAF 2.0 Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
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
plt.show()

erc_weights = optimization.get_erc_weights(extended_sigma)
df = pd.DataFrame({'Asset Class': asset_names, 'Weight': erc_weights})

plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('ERC Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
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

caaf = CAAF(expected_returns=expected_returns,
     covariance_matrix=covariance_matrix, target_md=0.39)
caam_weights = caaf.create_portfolio()

df = pd.DataFrame(caam_weights).reset_index()
df.columns = ['Asset Class', 'Weight']

plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('CAAF 1.0 Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.show()














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

    if covar_method == "constant":
        stds = np.sqrt(np.diag(covar))
    else:
        stds = returns.std() * np.sqrt(periodicity)
    # arithmetic_mu = exp_rets + np.square(stds) / 2
    arithmetic_mu = exp_rets

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

    # arithmetic_mu.loc['Asset Class 2'] = 0.13
    mv_return_optimum = minimize(
        return_objective,
        initial_weights,
        (-arithmetic_mu,),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-6}
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
        weight_objective, x0=initial_weights, args=(weight_ref, norm), method='SLSQP',
        constraints=constraints, bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    twostage_weights = pd.Series({exp_rets.index[i]: optimum.x[i] for i in range(n)})
    aligned_weights, aligned_mu = twostage_weights.align(arithmetic_mu, join='inner')
    # portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, covar)

    return twostage_weights






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

sigma = 0.07
x0 = np.full((len(selected_assets),), 1 / len(selected_assets))


rdf = pd.read_excel(f"./data/2024-08-31 master_file.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf = rdf.loc[rdf.loc[:, 'Date'] >= '1993-01-31', :]
rdf.set_index('Date', inplace=True)

# Calculate two-stage weights
weights = calc_two_stage_weights(
    rdf,
    extended_arithmetic_mu,
    target_volatility=sigma,
    epsilon=0.1,
    erc_weights=erc_weights,
    norm="l1",
    additional_constraints=None,
    covar_method="constant",
    periodicity=12,
    const_covar=covariance_matrix,
    initial_weights_two_stage=x0
)

# Convert to Series with asset names for consistency
two_stage_weights = pd.Series(weights, index=asset_names)

# Align the weights with `arithmetic_mu`
aligned_weights, aligned_mu = two_stage_weights.align(arithmetic_mu, join='inner')

# Calculate portfolio properties
#portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, extended_sigma)
#print(portfolio_properties)

# Plot Two-Stage Weights Portfolio
df = pd.DataFrame({'Asset Class': two_stage_weights.index, 'Weight': two_stage_weights.values})
plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('Two-Stage Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.show()




















































####################################################################################################
####################################################################################################
# Low Expected Return Example

target_vol = 0.07

# Asset 1, Asset 2, Equities, Credit, Government Bonds, Gold
asset_names = ["Asset Class 1", "Equity", "HY Credit", "Gov Bonds", "Gold"]
color_mapping = {
    "Equity": "rgb(64, 75, 151)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Asset Class 1": "rgb(160, 84, 66)",
    "HY Credit": "rgb(154, 183, 235)",
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
plt.title('CAAF 2.0 Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
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
plt.show()


erc_weights = optimization.get_erc_weights(extended_sigma)
df = pd.DataFrame({'Asset Class': asset_names, 'Weight': erc_weights})

plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('ERC Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
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
plt.show()

print("Break")

# Calculate two-stage weights
weights = calc_two_stage_weights(
    rdf,
    extended_arithmetic_mu,
    target_volatility=sigma,
    epsilon=0.1,
    erc_weights=erc_weights,
    norm="l1",
    additional_constraints=None,
    covar_method="constant",
    periodicity=12,
    const_covar=covariance_matrix,
    initial_weights_two_stage=x0
)

# Convert to Series with asset names for consistency
two_stage_weights = pd.Series(weights, index=asset_names)

# Align the weights with `arithmetic_mu`
aligned_weights, aligned_mu = two_stage_weights.align(arithmetic_mu, join='inner')

# Calculate portfolio properties
#portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, extended_sigma)
#print(portfolio_properties)

# Plot Two-Stage Weights Portfolio
df = pd.DataFrame({'Asset Class': two_stage_weights.index, 'Weight': two_stage_weights.values})
plt.figure(figsize=(10, 6))
sns.barplot(x='Asset Class', y='Weight', data=df, palette=color_mapping_hex)
plt.title('Two-Stage Portfolio')
plt.ylabel('Weight', fontsize=14)
plt.xlabel('Asset Class', fontsize=14)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(to_percent))
plt.show()
