import numpy as np
import pandas as pd

def calculate_tracking_error(weights, asset_covar, asset_benchmark_covar, benchmark_var):
    portfolio_variance = np.dot(np.dot(weights, asset_covar), weights)
    cross_term = np.dot(weights, asset_benchmark_covar)
    tracking_error = np.sqrt(portfolio_variance - 2 * cross_term + benchmark_var)
    return tracking_error

def tracking_error_constraint(weights, asset_covar, asset_benchmark_covar, benchmark_var, tracking_error_limit):
    tracking_error = calculate_tracking_error(weights, asset_covar, asset_benchmark_covar, benchmark_var)
    return tracking_error_limit - tracking_error

def read_benchmark_returns():
    benchmarks = pd.read_excel(f"./data/benchmarks.xlsx", sheet_name="Sheet1")
    benchmarks.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    benchmarks['Date'] = pd.to_datetime(benchmarks['Date'], format='%Y-%m-%d')
    benchmarks.set_index('Date', inplace=True)

    return benchmarks