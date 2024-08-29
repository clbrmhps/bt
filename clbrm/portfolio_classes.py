import numpy as np
import pandas as pd

from portfolio_construction import calculations, optimization
from scipy.optimize import minimize
from analysis.drawdowns import endpoint_mdd_lookup
from meucci.torsion import torsion
from meucci.EffectiveBets import EffectiveBets

def calculate_expected_return(weight, mu):
    return np.inner(weight, mu)

def calculate_variance(weight, cov):
    weight = weight.ravel()
    return np.einsum('i,ij,j->', weight, cov, weight)

def calculate_standard_deviation(weight, cov):
    weight = weight.ravel()
    return np.sqrt(np.einsum('i,ij,j->', weight, cov, weight))

def calculate_weight_distance(weight, weight_ref, norm, delta=0.5):
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

def scalar_effective_bets_wrapper(x, Sigma, t_MT):
    _, scalar_matrix = EffectiveBets(x, Sigma, t_MT)
    return -scalar_matrix.item()

class BasePortfolio():
    def __init__(self, tickers=None, bounds=None, constraints=None):
        self.n_assets = len(tickers)
        self.tickers = tickers
        self.bounds = bounds
        self.constraints = constraints

class ERC_CAAF(BasePortfolio):
    def __init__(self, covariance_matrix, constraints=None):
        self.asset_names = list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names, constraints=constraints)

        self.covariance_matrix = covariance_matrix.to_numpy()

    def create_portfolio(self):
        erc_weights = optimization.get_erc_weights(cov=self.covariance_matrix,
                                                   extra_constraints=self.constraints)

        weights = pd.Series(data=erc_weights, index=self.asset_names)

        return weights

class MeanVariance_CAAF(BasePortfolio):
    def __init__(self, expected_returns, covariance_matrix, constraints=None,
                 target_md=0.4):
        self.asset_names = list(expected_returns.index)

        super().__init__(tickers=self.asset_names, constraints=constraints)

        self.expected_returns = expected_returns.to_numpy()
        self.covariance_matrix = covariance_matrix.to_numpy()
        self.target_md = target_md

    def create_portfolio(self):
        mv_frontier = optimization.get_mv_frontier(mu=self.expected_returns,
                                                   cov=self.covariance_matrix,
                                                   query_points=1000,
                                                   target_md=self.target_md,
                                                   extra_constraints=self.constraints)

        mv_weights = mv_frontier['Optimal Portfolio Weights']

        weights = pd.Series(data=mv_weights, index=self.asset_names)

        return weights

class MeanVariance(BasePortfolio):
    def __init__(self, expected_returns, covariance_matrix, constraints=[],
                 target_volatility=0.07):
        self.asset_names = list(expected_returns.index)

        super().__init__(tickers=self.asset_names, constraints=constraints)

        self.expected_returns = expected_returns.to_numpy()
        self.covariance_matrix = covariance_matrix.to_numpy()
        self.target_volatility = target_volatility
        self.equal_weights = np.repeat(1/self.n_assets, self.n_assets)
        self.bounds = [(0, 1) for _ in range(self.n_assets)]
        self.constraints = constraints

    def create_portfolio(self):
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(np.dot(w, self.covariance_matrix), w)) - self.target_volatility}
        ]
        constraints += self.constraints

        result = minimize(
                 lambda w: -calculate_expected_return(w, self.expected_returns),  # We minimize the negative return to maximize the return
                 self.equal_weights,
                 method='SLSQP',
                 bounds=self.bounds,
                 constraints=constraints,
                 options={'maxiter': 10_000, 'ftol': 1e-15}
        )

        mv_weights = result.x
        weights = pd.Series(data=mv_weights, index=self.asset_names)

        return weights


class CAAF(BasePortfolio):
    def __init__(self, expected_returns, covariance_matrix, constraints=None,
                 target_md=0.4):
        if isinstance(expected_returns, pd.Series):
            self.asset_names = expected_returns.index.tolist()
        elif isinstance(expected_returns, np.ndarray):
            self.asset_names = [f'Asset_{i}' for i in range(len(expected_returns))]
        else:
            raise ValueError('expected_returns must be either a pandas Series or a numpy array.')

        assert self.asset_names == list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names, constraints=constraints)

        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix

        self.constraints = constraints

        self.target_md = target_md

    def create_portfolio(self):
        if self.constraints:
            hy_weight = self.constraints[0]
            alternatives_weight = self.constraints[1]

            erc_extra_constraints = [
                {'type': 'ineq', 'fun': lambda w: (alternatives_weight - w[5]), 'name': 'Alternatives Constraint'},
                {'type': 'ineq', 'fun': lambda w: (hy_weight - w[2]), 'name': 'HY Credit Constraint'}]
        else:
            erc_extra_constraints = None

        erc_weights = ERC_CAAF(self.covariance_matrix,
                               erc_extra_constraints).create_portfolio()

        if self.constraints:
            em_percentage = self.constraints[2]
            mv_extra_constraints = [
                {'type': 'ineq', 'fun': lambda w: (w[0] + w[1] + erc_weights[0] + erc_weights[1]) * em_percentage -
                                                  w[1] - erc_weights[1], 'name': 'EM Equities Constraint'},
                {'type': 'ineq', 'fun': lambda w: (alternatives_weight * 2 - w[5] - erc_weights[5]),
                 'name': 'Alternatives Constraint'},
                {'type': 'ineq', 'fun': lambda w: (hy_weight * 2 - w[2] - erc_weights[2]),
                 'name': 'HY Credit Constraint'}]
        else:
            mv_extra_constraints = None

        mv_weights = MeanVariance_CAAF(self.expected_returns, self.covariance_matrix,
                                       mv_extra_constraints, target_md=self.target_md).create_portfolio()

        caam_weights = erc_weights * 0.5 + mv_weights * 0.5

        weights = pd.Series(data=caam_weights, index=self.asset_names)

        return weights

class MaximumDiversification(BasePortfolio):
    def __init__(self, expected_returns, covariance_matrix, target_volatility=None, epsilon=None, penalty_lambda=None,
                 constraints=[]):
        if isinstance(expected_returns, pd.Series):
            self.asset_names = expected_returns.index.tolist()
        elif isinstance(expected_returns, np.ndarray):
            self.asset_names = [f'Asset_{i}' for i in range(len(expected_returns))]
        else:
            raise ValueError('expected_returns must be either a pandas Series or a numpy array.')

        assert self.asset_names == list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names)

        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.constraints = constraints

        self.equal_weights = np.repeat(1/self.n_assets, self.n_assets)
        self.bounds = [(0, 1) for _ in range(self.n_assets)]
        self.target_volatility = target_volatility
        self.epsilon = epsilon
        self.penalty_lambda = penalty_lambda

        self.minimum_variance_weights = MinimumVariance(self.covariance_matrix).create_portfolio()

        self.minimum_minvar_expected_returns = calculate_expected_return(self.minimum_variance_weights, self.expected_returns)
        self.minimum_minvar_standard_deviation = calculate_standard_deviation(self.minimum_variance_weights, self.covariance_matrix)

        self.maximum_minvar_expected_returns = np.max(self.expected_returns)
        self.maximum_minvar_standard_deviation = np.max(np.sqrt(np.diag(self.covariance_matrix)))


    def maximum_diversification_objective(self, w, *args):
        sigma, t_MT, weight_ref, norm, penalty_coeff = args
        original_objective = scalar_effective_bets_wrapper(w, sigma, t_MT)
        penalty_term = penalty_coeff * calculate_weight_distance(w, weight_ref, norm)
        return original_objective + penalty_term

    def create_portfolio(self, reference_weight=None, target_volatility=None):
        if reference_weight is None:
            reference_weight = np.repeat(1/self.n_assets, self.n_assets)
        if isinstance(self.constraints, list) and len(self.constraints) == 3:
            em_percentage = self.constraints[2]
            alternatives_weight = self.constraints[1]
            hy_weight = self.constraints[0]

            extra_constraints = [
                 {'type': 'ineq', 'fun': lambda w: (w[0] + w[1])  * em_percentage -
                                                   w[1], 'name': 'EM Equities Constraint'},
                 {'type': 'ineq', 'fun': lambda w: alternatives_weight - w[5],
                  'name': 'Alternatives Constraint'},
                 {'type': 'ineq', 'fun': lambda w: hy_weight - w[2],
                  'name': 'HY Credit Constraint'}]
        else:
            extra_constraints = []

        mv_weights = MeanVariance(self.expected_returns,
                                  self.covariance_matrix,
                                  constraints=extra_constraints,
                                  target_volatility=target_volatility).create_portfolio()
        target_return = calculate_expected_return(mv_weights, self.expected_returns)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(np.dot(w, self.covariance_matrix), w)) - target_volatility},
            {'type': 'ineq', 'fun': lambda w: calculate_expected_return(w, self.expected_returns) - (1 - self.epsilon) * target_return}
        ]

        constraints += extra_constraints

        t_MT = torsion(self.covariance_matrix, 'minimum-torsion')

        result = minimize(
            self.maximum_diversification_objective,
            self.minimum_variance_weights,
            args=(self.covariance_matrix.to_numpy(),
                  t_MT,
                  reference_weight.to_numpy(),
                  "huber", self.penalty_lambda),
            method='SLSQP',
            bounds=self.bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        weights = pd.Series({self.asset_names[i]: result.x[i] for i in range(self.n_assets)})

        return weights

    def create_mean_variance_frontier(self):
        target_returns = np.linspace(self.minimum_minvar_expected_returns,
                                     self.maximum_minvar_standard_deviation, 100)

        mv_frontier = pd.DataFrame()

        for target_return in target_returns:
          constraints = (
              {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
              {'type': 'eq', 'fun': lambda w: calculate_expected_return(w, self.expected_returns) - target_return}
          )

          result = minimize(
              lambda w: calculate_variance(w, self.covariance_matrix),
              self.equal_weights,
              method='SLSQP',
              bounds=self.bounds,
              constraints=constraints,
              options={'maxiter': 10000, 'ftol': 1e-15}
          )

          if result.success:
              portfolio_variance_var = result.fun
              portfolio_std_dev = np.sqrt(portfolio_variance_var)
              new_row = {'Target Return': target_return, 'Portfolio Variance': portfolio_variance_var,
                         'Portfolio Std Dev': portfolio_std_dev}
              mv_frontier = mv_frontier.append(new_row, ignore_index=True)

        return mv_frontier

    def create_frontier(self, stdev_grid_size=0.0025):
        target_stdevs = np.arange(self.minimum_minvar_standard_deviation,
                                  self.maximum_minvar_standard_deviation, stdev_grid_size)

        self.frontier_df = pd.DataFrame()

        reference_weights = self.minimum_variance_weights
        for target_stdev in target_stdevs:
            target_volatility = target_stdev
            max_enb_weights = self.create_portfolio(reference_weights, target_volatility)
            reference_weights = max_enb_weights

            max_enb_expected_return = calculate_expected_return(max_enb_weights, self.expected_returns)
            max_enb_standard_deviation = calculate_standard_deviation(max_enb_weights, self.covariance_matrix)

            row = {
                'max_enb_expected_return': max_enb_expected_return,
                'max_enb_standard_deviation': max_enb_standard_deviation,
                'max_enb_weights': [max_enb_weights]
            }

            self.frontier_df = self.frontier_df.append(row, ignore_index=True)

        return self.frontier_df

    def select_from_frontier(self, selected_standard_deviation):

        self.frontier_df['abs_difference'] = np.abs(
            self.frontier_df['max_enb_standard_deviation'] - selected_standard_deviation)
        min_diff_row_index = self.frontier_df['abs_difference'].idxmin()
        closest_row = self.frontier_df.loc[min_diff_row_index]
        self.frontier_df.drop('abs_difference', axis=1, inplace=True)

        return closest_row


class MinimumVariance(BasePortfolio):
    def __init__(self, covariance_matrix):
        self.asset_names = list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names)

        self.covariance_matrix = covariance_matrix.to_numpy()

        self.equal_weight = np.repeat(1/self.n_assets, self.n_assets)
        self.bounds = [(0, 1) for _ in range(self.n_assets)]

    def calculate_minimum_variance_portfolio(self, cov_matrix):
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        weights = np.dot(inv_cov_matrix, ones) / np.dot(ones.T, np.dot(inv_cov_matrix, ones))
        return weights

    def create_portfolio(self):
        min_var_weights_analytical = self.calculate_minimum_variance_portfolio(self.covariance_matrix)

        weights = pd.Series(data=min_var_weights_analytical, index=self.asset_names)

        return weights

