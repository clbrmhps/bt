import cvxpy

import cvxpy as cp
import numpy as np

from scipy.linalg import sqrtm

exp_rets.dropna(inplace=True)
n = len(exp_rets)

exp_rets_np = exp_rets.values

# Define the weights as a CVXPY variable
weights = cp.Variable(n)

target_volatility_param = cp.Parameter(nonneg=True)
target_volatility_param.value = target_volatility
# Define the return objective function
def return_objective_cvxpy(exp_rets, weights):
    return cp.sum(cp.multiply(exp_rets, weights))

# Define the volatility constraint function
def volatility_constraint_cvxpy(weights, covar, target_volatility):
    return cp.sqrt(cp.quad_form(weights, covar)) - target_volatility

# Define the weight objective function using CVXPY norms
def weight_objective_cvxpy(weights, weight_ref, norm):
    if norm == 'l1':
        return cp.norm(weights - weight_ref, 1)
    elif norm == 'l2':
        return cp.norm(weights - weight_ref, 2)


portfolio_variance = cp.quad_form(weights, covar.values)

# Tolerance for the equality constraint
tolerance = 1e-6

variance_upper_bound = target_volatility_param**2 + tolerance
variance_lower_bound = target_volatility_param**2 - tolerance

volatility_constraints = [
    portfolio_variance <= variance_upper_bound,
    portfolio_variance >= variance_lower_bound
]

# Now include this constraint in the list of constraints for the optimization problem
constraints = [cp.sum(weights) == 1] + volatility_constraint
constraints += [weights >= 0, weights <= 1]

portfolio_return = exp_rets_np @ weights

# First optimization stage (maximize returns)
problem = cp.Problem(cp.Maximize(portfolio_return), constraints)
problem.solve()

# Check if the problem was successfully solved
if problem.status not in ["optimal", "optimal_inaccurate"]:
    raise Exception("The optimization problem did not converge.")

# Extract the target return value
target_return = problem.value

# Second optimization stage (minimize norm difference)
constraints.append(exp_rets_np @ weights >= (1 - epsilon) * target_return)
problem = cp.Problem(cp.Minimize(cp.norm(weights-weight_ref)), constraints)
problem.solve()

# Check if the problem was successfully solved
if problem.status not in ["optimal", "optimal_inaccurate"]:
    raise Exception("The optimization problem did not converge.")

# Extract the optimized weights
optimized_weights = weights.value
twostage_weights = pd.Series({exp_rets.index[i]: optimized_weights[i] for i in range(n)})
aligned_weights, aligned_mu = twostage_weights.align(arithmetic_mu, join='inner')
portfolio_properties = calculate_portfolio_properties(aligned_weights, aligned_mu, covar)
