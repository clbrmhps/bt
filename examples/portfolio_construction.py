import numpy as np
import pandas as pd

from examples.base_portfolio import BasePortfolio
from portfolio_construction import calculations, optimization
from scipy.optimize import minimize
from analysis.drawdowns import endpoint_mdd_lookup
from meucci.torsion import torsion
from meucci.EffectiveBets import EffectiveBets

def calculate_expected_return(weight, mu):
    return np.array([np.inner(weight, mu)])

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

class CAAF(BasePortfolio):
    def __init__(self, expected_returns, covariance_matrix, constraints=None,
                 target_md=0.3):
        if isinstance(expected_returns, pd.Series):
            self.asset_names = expected_returns.index.tolist()
        elif isinstance(expected_returns, np.ndarray):
            self.asset_names = [f'Asset_{i}' for i in range(len(expected_returns))]
        else:
            raise ValueError('expected_returns must be either a pandas Series or a numpy array.')

        assert self.asset_names == list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names)

        self.expected_returns = expected_returns.to_numpy()
        self.covariance_matrix = covariance_matrix.to_numpy()
        self.constraints = constraints

        self.target_md = target_md

    def create_portfolio(self):
        erc_weights = optimization.get_erc_weights(self.covariance_matrix)
        mv_frontier = optimization.get_mv_frontier(mu=self.expected_returns,
                                                   cov=self.covariance_matrix,
                                                   query_points=1000,
                                                   target_md=self.target_md,
                                                   extra_constraints=self.constraints)

        mv_weights = mv_frontier['Optimal Portfolio Weights']
        caam_weights = erc_weights * 0.5 + mv_weights * 0.5

        weights = pd.Series(data=caam_weights, index=self.asset_names)

        return weights


class MaximumDiversification(BasePortfolio):
    def __init__(self, expected_returns, covariance_matrix, target_volatility, epsilon,
                 constraints=None):
        if isinstance(expected_returns, pd.Series):
            self.asset_names = expected_returns.index.tolist()
        elif isinstance(expected_returns, np.ndarray):
            self.asset_names = [f'Asset_{i}' for i in range(len(expected_returns))]
        else:
            raise ValueError('expected_returns must be either a pandas Series or a numpy array.')

        assert self.asset_names == list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names)

        self.expected_returns = expected_returns.to_numpy()
        self.covariance_matrix = covariance_matrix.to_numpy()
        self.constraints = constraints

        self.equal_weights = np.repeat(1/self.n_assets, self.n_assets)
        self.bounds = [(0, 1) for _ in range(self.n_assets)]
        self.target_volatility = target_volatility
        self.epsilon = epsilon

        self.minimum_variance_weights = MinimumVariance(self.covariance_matrix).create_portfolio()

    def maximum_diversification_objective(w, *args):
        sigma, t_MT, weight_ref, norm, penalty_coeff = args
        original_objective = scalar_effective_bets_wrapper(w, sigma, t_MT)
        penalty_term = penalty_coeff * calculate_weight_distance(w, weight_ref, norm)
        return original_objective + penalty_term

    def create_portfolio(self):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(np.dot(w, self.covariance_matrix), w)) - self.target_volatility}
        )

        result = minimize(
            lambda w: -calculate_expected_return(w, self.expected_returns),  # We minimize the negative return to maximize the return
            self.equal_weights,
            method='SLSQP',
            bounds=self.bounds,
            constraints=constraints,
            options={'maxiter': 10_000, 'ftol': 1e-15}
        )

        print(result.x)
        target_return = -result.fun
        mv_weights = pd.Series({self.asset_names[i]: result.x[i] for i in range(self.n_assets)})

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(np.dot(w, self.covariance_matrix), w)) - self.target_volatility},
            {'type': 'ineq', 'fun': lambda w: calculate_expected_return(w, self.expected_returns) - (1 - self.epsilon) * target_return}
        )

        t_MT = torsion(self.covariance_matrix, 'minimum-torsion')

        result = minimize(
            self.maximum_diversification_objective,
            self.minimum_variance_weights,
            args=(self.covariance_matrix, t_MT, weight_ref, "huber", penalty_coeff),
            method='SLSQP',
            bounds=extended_bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        return weights


class MinimumVariance(BasePortfolio):
    def __init__(self, covariance_matrix):
        self.asset_names = list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names)

        self.covariance_matrix = covariance_matrix.to_numpy()

        self.equal_weight = np.repeat(1/self.n_assets, self.n_assets)
        self.bounds = [(0, 1) for _ in range(self.n_assets)]

    def calculate_minimum_variance_portfolio(cov_matrix):
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        weights = np.dot(inv_cov_matrix, ones) / np.dot(ones.T, np.dot(inv_cov_matrix, ones))
        return weights

    def create_portfolio(self):
        min_var_weights_analytical = self.calculate_minimum_variance_portfolio(self.covariance_matrix)

        weights = pd.Series(data=min_var_weights_analytical, index=self.asset_names)

        return weights

class ERC(BasePortfolio):
    def __init__(self, covariance_matrix):
        self.asset_names = list(covariance_matrix.columns)

        super().__init__(tickers=self.asset_names)

        self.covariance_matrix = covariance_matrix.to_numpy()

    def create_portfolio(self):
        erc_weights = optimization.get_erc_weights(self.covariance_matrix)

        weights = pd.Series(data=erc_weights, index=self.asset_names)

        return weights