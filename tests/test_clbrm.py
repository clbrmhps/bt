import numpy as np
import pytest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from clbrm.portfolio_classes import calculate_expected_return
from clbrm.portfolio_classes import calculate_standard_deviation

from clbrm.portfolio_classes import ERC_CAAF
from clbrm.portfolio_classes import MeanVariance_CAAF
from clbrm.portfolio_classes import CAAF
from clbrm.portfolio_classes import MaximumDiversification
from clbrm.portfolio_classes import MeanVariance

color_mapping = {
    "DM Equities": "rgb(64, 75, 151)",
    "EM Equities": "rgb(131, 140, 202)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)",
    "Alternatives": "rgb(160, 84, 66)",
}

color_mapping_bardeen = {
    "DM Equities": "rgb(64, 75, 151)",
    "EM Equities": "rgb(131, 140, 202)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)",
    "Alternatives": "rgb(160, 84, 66)",
    "PID": "rgb(189, 181, 19)",
    "RED": "rgb(144, 121, 65)"
}

def plot_weights_with_mapping(series, color_mapping):
    """
    Plot a Pandas Series as a bar chart using Seaborn with specified colors.

    Parameters:
    - series: Pandas Series to plot.
    - color_mapping: Dictionary mapping series index to colors in RGB or Hex format.
    """
    # Convert RGB colors to Hex if not already in Hex
    color_mapping_hex = {
        asset: color if color.startswith('#') else '#' + ''.join(
            f'{int(c):02x}' for c in color.strip('rgb()').split(', '))
        for asset, color in color_mapping.items()
    }

    # Convert the Series to a DataFrame
    df = series.reset_index()
    df.columns = ['Asset Class', 'Weight']

    # Map colors to the DataFrame's categories
    palette = df['Asset Class'].map(color_mapping_hex).tolist()

    # Plotting
    sns.barplot(x='Asset Class', y='Weight', data=df, palette=palette)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # 1 means 100% is 1 in data units
    plt.ylim(0, 0.6)
    plt.xticks(rotation=45)  # Rotate category labels for better readability
    plt.tight_layout()
    plt.show()

# Define a fixture that loads the covariance matrix
@pytest.fixture
def covariance_matrix():
    return np.load('./data/covariance_matrix_production_20240315.npy')

@pytest.fixture
def expected_returns():
    return np.load('./data/expected_returns_production_20240315.npy')

@pytest.fixture
def covariance_matrix_bardeen():
    return np.load('./data/covariance_matrix_bardeen_20240229.npy')

@pytest.fixture
def expected_returns_bardeen():
    return np.load('./data/expected_returns_bardeen_20240229.npy')

# def test_erc_weights_production(covariance_matrix):
#     asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
#     covariance_matrix = pd.DataFrame(covariance_matrix, columns=asset_names, index=asset_names)
#
#     alternatives_weight = 0.165489404641776
#     hy_weight = 0.07686384992791485
#
#     erc_extra_constraints = [
#         {'type': 'ineq', 'fun': lambda w: (alternatives_weight - w[5]), 'name': 'Alternatives Constraint'},
#         {'type': 'ineq', 'fun': lambda w: (hy_weight - w[2]), 'name': 'HY Credit Constraint'}]
#
#     erc_class = ERC_CAAF(covariance_matrix, erc_extra_constraints)
#     erc_weights = erc_class.create_portfolio()
#
#     # Expected weights
#     expected_weights = pd.Series({
#         'DM Equities': 0.096336,
#         'EM Equities': 0.070110,
#         'HY Credit': 0.076864,
#         'Gov Bonds': 0.485100,
#         'Gold': 0.106100,
#         'Alternatives': 0.165489,
#     })
#
#     # Check each expected weight against the actual weight
#     for asset, expected_weight in expected_weights.items():
#         assert erc_weights[asset] == pytest.approx(expected_weight, rel=1e-5), f"Weight for {asset} did not match expected value."
#
#
# def test_mv_weights_production(expected_returns, covariance_matrix, target_md=0.418):
#     asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
#     covariance_matrix = pd.DataFrame(covariance_matrix, columns=asset_names, index=asset_names)
#     expected_returns = pd.Series(expected_returns, index=asset_names)
#
#     alternatives_weight = 0.165489404641776
#     hy_weight = 0.07686384992791485
#
#     em_percentage = 0.3
#
#     erc_weights = np.array([0.09633624, 0.07011023, 0.07686385, 0.48509992, 0.10610036,
#        0.1654894])
#
#     mv_extra_constraints = [
#         {'type': 'ineq', 'fun': lambda w: (w[0] + w[1] + erc_weights[0] + erc_weights[1]) * em_percentage -
#                                           w[1] - erc_weights[1], 'name': 'EM Equities Constraint'},
#         {'type': 'ineq', 'fun': lambda w: (alternatives_weight * 2 - w[5] - erc_weights[5]),
#          'name': 'Alternatives Constraint'},
#         {'type': 'ineq', 'fun': lambda w: (hy_weight * 2 - w[2] - erc_weights[2]),
#          'name': 'HY Credit Constraint'}]
#
#     mv_class = MeanVariance_CAAF(expected_returns, covariance_matrix, mv_extra_constraints, target_md)
#     mv_weights = mv_class.create_portfolio()
#
#     # Expected weights
#     expected_weights = pd.Series({
#           'DM Equities': 0.517482,
#           'EM Equities': 0.192955,
#           'HY Credit': 0.076864,
#           'Gov Bonds': 0.000000,
#           'Gold': 0.047210,
#           'Alternatives': 0.165489,
#      })
#
#     # Check each expected weight against the actual weight
#     for asset, expected_weight in expected_weights.items():
#         assert mv_weights[asset] == pytest.approx(expected_weight, rel=1e-5), f"Weight for {asset} did not match expected value."
#

def test_caam_weights_production(expected_returns, covariance_matrix, target_md=0.418):
    asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
    covariance_matrix = pd.DataFrame(covariance_matrix, columns=asset_names, index=asset_names)
    expected_returns = pd.Series(expected_returns, index=asset_names)

    alternatives_weight = 0.165489404641776
    hy_weight = 0.07686384992791485

    em_percentage = 0.3

    constraints = [hy_weight, alternatives_weight, em_percentage]

    caaf_class = CAAF(expected_returns, covariance_matrix, constraints, target_md)
    caaf_weights = caaf_class.create_portfolio()

    # Expected weights
    expected_weights = pd.Series({
        'DM Equities': 0.306909,
        'EM Equities': 0.131532,
        'HY Credit': 0.076864,
        'Gov Bonds': 0.242550,
        'Gold': 0.076655,
        'Alternatives': 0.165489,
    })

    # Check each expected weight against the actual weight
    for asset, expected_weight in expected_weights.items():
        assert caaf_weights[asset] == pytest.approx(expected_weight,
                                                  rel=1e-5), f"Weight for {asset} did not match expected value."

    plot_weights_with_mapping(caaf_weights, color_mapping)

# def test_caam_weights_unconstrained(expected_returns, covariance_matrix, target_md=0.418):
#     asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
#     covariance_matrix = pd.DataFrame(covariance_matrix, columns=asset_names, index=asset_names)
#     expected_returns = pd.Series(expected_returns, index=asset_names)
#
#     caaf_class = CAAF(expected_returns, covariance_matrix, target_md=target_md)
#     caaf_weights = caaf_class.create_portfolio()
#
#     # Expected weights
#     expected_weights = pd.Series({
#         'DM Equities': 0.036702,
#         'EM Equities': 0.304829,
#         'HY Credit': 0.066291,
#         'Gov Bonds': 0.185424,
#         'Gold': 0.044358,
#         'Alternatives': 0.362396,
#     })
#
#     # Check each expected weight against the actual weight
#     for asset, expected_weight in expected_weights.items():
#         assert caaf_weights[asset] == pytest.approx(expected_weight,
#                                                   rel=1e-4), f"Weight for {asset} did not match expected value."
#
#     plot_weights_with_mapping(caaf_weights, color_mapping)

# def test_maximum_diversification_weights_unconstrained(expected_returns, covariance_matrix, target_volatility=0.07):
#         asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
#         covariance_matrix = pd.DataFrame(covariance_matrix, columns=asset_names, index=asset_names)
#         expected_returns = pd.Series(expected_returns, index=asset_names)
#
#         maxdiv_class = MaximumDiversification(expected_returns=expected_returns,
#                                               covariance_matrix=covariance_matrix,
#                                               epsilon=0.1,
#                                               penalty_lambda=20)
#         frontier_df = maxdiv_class.create_frontier(stdev_grid_size=0.001)
#         selected_stdev = maxdiv_class.select_from_frontier(0.078)
#
#         selected_weights = selected_stdev['max_enb_weights'][0]
#
#         # Expected weights
#         expected_weights = pd.Series({
#             'DM Equities': 0.159430,
#             'EM Equities': 0.197038,
#             'HY Credit': 0.017340,
#             'Gov Bonds': 0.000000,
#             'Gold': 0.140567,
#             'Alternatives': 0.485625,
#         })
#
#         # Check each expected weight against the actual weight
#         for asset, expected_weight in expected_weights.items():
#             assert selected_weights[asset] == pytest.approx(expected_weight,
#                                                       rel=1e-4), f"Weight for {asset} did not match expected value."
#
#         plot_weights_with_mapping(selected_weights, color_mapping)
#
#         target_stdevs = frontier_df.loc[:, "max_enb_standard_deviation"]
#
#         mv_df = pd.DataFrame()
#         for target_stdev in target_stdevs:
#             mv_weights = MeanVariance(expected_returns,
#                          covariance_matrix,
#                          target_volatility=target_stdev).create_portfolio()
#
#             mv_expected_return = calculate_expected_return(mv_weights, expected_returns)
#             mv_df = mv_df.append({'Standard Deviation': target_stdev, 'Expected Return': mv_expected_return},
#                                  ignore_index=True)
#
#         sns.scatterplot(data=frontier_df, x='max_enb_standard_deviation', y='max_enb_expected_return', label='Frontier')
#         sns.scatterplot(data=mv_df, x='Standard Deviation', y='Expected Return', color='red', label='Mean-Variance')
#
#         plt.title('Expected Return vs Standard Deviation')
#         plt.xlabel('Standard Deviation')
#         plt.ylabel('Expected Return')
#         plt.legend()
#         plt.show()

# def test_maximum_diversification_weights_production(expected_returns, covariance_matrix, target_volatility=0.07):
#     asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
#     covariance_matrix = pd.DataFrame(covariance_matrix, columns=asset_names, index=asset_names)
#     expected_returns = pd.Series(expected_returns, index=asset_names)
#
#     alternatives_weight = 0.165489404641776
#     hy_weight = 0.07686384992791485
#
#     em_percentage = 0.3
#
#     constraints = [hy_weight, alternatives_weight, em_percentage]
#
#     maxdiv_class = MaximumDiversification(expected_returns=expected_returns,
#                                           covariance_matrix=covariance_matrix,
#                                           constraints=constraints,
#                                           epsilon=0.05,
#                                           penalty_lambda=20)
#     frontier_df = maxdiv_class.create_frontier(stdev_grid_size=0.001)
#     selected_stdev = maxdiv_class.select_from_frontier(0.078)
#     selected_weights = selected_stdev['max_enb_weights'][0]
#
#     # Expected weights
#     expected_weights = pd.Series({
#         'DM Equities': 0.261204,
#         'EM Equities': 0.111945,
#         'HY Credit':  0.076864,
#         'Gov Bonds': 0.173427,
#         'Gold': 0.211072,
#         'Alternatives': 0.165489,
#     })
#
#     # Check each expected weight against the actual weight
#     for asset, expected_weight in expected_weights.items():
#         assert selected_weights[asset] == pytest.approx(expected_weight,
#                                                         rel=1e-4), f"Weight for {asset} did not match expected value."
#
#     plot_weights_with_mapping(selected_weights, color_mapping)
#
#     expected_return_arithmetic = calculate_expected_return(selected_weights, expected_returns)
#     standard_deviation = calculate_standard_deviation(selected_weights, covariance_matrix)
#     expected_return_geometric = expected_return_arithmetic - standard_deviation**2 / 2
#
#     if isinstance(constraints, list) and len(constraints) == 3:
#         em_percentage = constraints[2]
#         alternatives_weight = constraints[1]
#         hy_weight = constraints[0]
#
#         extra_constraints = [
#             {'type': 'ineq', 'fun': lambda w: (w[0] + w[1]) * em_percentage -
#                                               w[1], 'name': 'EM Equities Constraint'},
#             {'type': 'ineq', 'fun': lambda w: alternatives_weight - w[5],
#              'name': 'Alternatives Constraint'},
#             {'type': 'ineq', 'fun': lambda w: hy_weight - w[2],
#              'name': 'HY Credit Constraint'}]
#
#     target_stdevs = frontier_df.loc[:, "max_enb_standard_deviation"]
#     mv_df = pd.DataFrame()
#     for target_stdev in target_stdevs:
#         mv_weights = MeanVariance(expected_returns,
#                                   covariance_matrix,
#                                   constraints=extra_constraints,
#                                   target_volatility=target_stdev).create_portfolio()
#
#         mv_expected_return = calculate_expected_return(mv_weights, expected_returns)
#         mv_df = mv_df.append({'Standard Deviation': target_stdev, 'Expected Return': mv_expected_return},
#                              ignore_index=True)
#
#     frontier_df_plot = frontier_df[frontier_df['max_enb_standard_deviation'] > 0.039]
#     mv_df_plot = mv_df[mv_df['Standard Deviation'] > 0.039]
#     frontier_df_plot = frontier_df_plot[frontier_df_plot['max_enb_standard_deviation'] < 0.155]
#     mv_df_plot = mv_df_plot[mv_df_plot['Standard Deviation'] < 0.155]
#
#     sns.scatterplot(data=frontier_df_plot, x='max_enb_standard_deviation', y='max_enb_expected_return', label='Frontier')
#     sns.scatterplot(data=mv_df_plot, x='Standard Deviation', y='Expected Return', color='red', label='Mean-Variance')
#
#     plt.title('Expected Return vs Standard Deviation')
#     plt.xlabel('Standard Deviation')
#     plt.ylabel('Expected Return')
#     plt.legend()
#     plt.show()

def test_caam_weights_bardeen_production(expected_returns_bardeen, covariance_matrix_bardeen, target_md=0.418):
    asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives', 'PID', 'RED']
    covariance_matrix = pd.DataFrame(covariance_matrix_bardeen, columns=asset_names, index=asset_names)
    expected_returns = pd.Series(expected_returns_bardeen, index=asset_names)

    alternatives_weight = 0.1648 * 2342 / 4900
    hy_weight = 0.087 * 2342 / 4900

    em_percentage = 0.3

    constraints = [hy_weight, alternatives_weight, em_percentage]

    caaf_class = CAAF(expected_returns, covariance_matrix, constraints, target_md)
    caaf_weights = caaf_class.create_portfolio()

    # Expected weights
    expected_weights = pd.Series({
        'DM Equities': 0.274095,
        'EM Equities': 0.117469,
        'HY Credit': 0.020791,
        'Gov Bonds': 0.218940,
        'Gold': 0.044423,
        'Alternatives': 0.039384,
        'PID': 0.041557,
        'RED': 0.243341
    })

    # Check each expected weight against the actual weight
    for asset, expected_weight in expected_weights.items():
        assert caaf_weights[asset] == pytest.approx(expected_weight,
                                                  rel=1e-4), f"Weight for {asset} did not match expected value."

    plot_weights_with_mapping(caaf_weights, color_mapping_bardeen)

def test_maximum_diversification_weights_bardeen_production(expected_returns_bardeen, covariance_matrix_bardeen, target_volatility=0.07):
    asset_names = ['DM Equities', 'EM Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives', 'PID', 'RED']
    covariance_matrix = pd.DataFrame(covariance_matrix_bardeen, columns=asset_names, index=asset_names)
    expected_returns = pd.Series(expected_returns_bardeen, index=asset_names)

    alternatives_weight = 0.1648 * 2342 / 4900
    hy_weight = 0.087 * 2342 / 4900

    em_percentage = 0.3

    constraints = [hy_weight, alternatives_weight, em_percentage]

    maxdiv_class = MaximumDiversification(expected_returns=expected_returns,
                                          covariance_matrix=covariance_matrix,
                                          constraints=constraints,
                                          epsilon=0.05,
                                          penalty_lambda=20)
    frontier_df = maxdiv_class.create_frontier(stdev_grid_size=0.001)
    selected_stdev = maxdiv_class.select_from_frontier(0.078)
    selected_weights = selected_stdev['max_enb_weights'][0]

    # Expected weights
    expected_weights = pd.Series({
        'DM Equities': 0.157239,
        'EM Equities': 0.067388,
        'HY Credit':  0.041582,
        'Gov Bonds': 0.036490,
        'Gold': 0.080284,
        'Alternatives': 0.078768,
        'PID': 0.124577,
        'RED': 0.413673
    })

    # Check each expected weight against the actual weight
    for asset, expected_weight in expected_weights.items():
        assert selected_weights[asset] == pytest.approx(expected_weight,
                                                        rel=1e-4), f"Weight for {asset} did not match expected value."

    plot_weights_with_mapping(selected_weights, color_mapping_bardeen)

    expected_return_arithmetic = calculate_expected_return(selected_weights, expected_returns)
    standard_deviation = calculate_standard_deviation(selected_weights, covariance_matrix)
    expected_return_geometric = expected_return_arithmetic - standard_deviation**2 / 2

    if isinstance(constraints, list) and len(constraints) == 3:
        em_percentage = constraints[2]
        alternatives_weight = constraints[1]
        hy_weight = constraints[0]

        extra_constraints = [
            {'type': 'ineq', 'fun': lambda w: (w[0] + w[1]) * em_percentage -
                                              w[1], 'name': 'EM Equities Constraint'},
            {'type': 'ineq', 'fun': lambda w: alternatives_weight - w[5],
             'name': 'Alternatives Constraint'},
            {'type': 'ineq', 'fun': lambda w: hy_weight - w[2],
             'name': 'HY Credit Constraint'}]

    target_stdevs = frontier_df.loc[:, "max_enb_standard_deviation"]
    mv_df = pd.DataFrame()
    for target_stdev in target_stdevs:
        mv_weights = MeanVariance(expected_returns,
                                  covariance_matrix,
                                  constraints=extra_constraints,
                                  target_volatility=target_stdev).create_portfolio()

        mv_expected_return = calculate_expected_return(mv_weights, expected_returns)
        mv_df = mv_df.append({'Standard Deviation': target_stdev, 'Expected Return': mv_expected_return},
                             ignore_index=True)

    # frontier_df_plot = frontier_df[frontier_df['max_enb_standard_deviation'] > 0.039]
    # mv_df_plot = mv_df[mv_df['Standard Deviation'] > 0.039]
    # frontier_df_plot = frontier_df_plot[frontier_df_plot['max_enb_standard_deviation'] < 0.155]
    # mv_df_plot = mv_df_plot[mv_df_plot['Standard Deviation'] < 0.155]

    sns.scatterplot(data=frontier_df, x='max_enb_standard_deviation', y='max_enb_expected_return', label='Frontier')
    sns.scatterplot(data=mv_df, x='Standard Deviation', y='Expected Return', color='red', label='Mean-Variance')

    plt.title('Expected Return vs Standard Deviation')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.show()
