import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

from cycler import cycler
def get_caaf_colors():
    return ((64 / 255, 75 / 255, 151 / 255),
            (131 / 255, 140 / 255, 202 / 255),
            (154 / 255, 183 / 255, 235 / 255),
            (144 / 255, 143 / 255, 74 / 255),
            (216 / 255, 169 / 255, 23 / 255),
            (160 / 255, 84 / 255, 66 / 255),
            (189 / 255, 181 / 255, 19 / 255),
            (144 / 255, 121 / 255, 65 / 255))
mpl.rcParams['axes.prop_cycle'] = cycler(color=get_caaf_colors())
mpl.rcParams['grid.color'] = 'lightgrey'
mpl.rcParams['grid.linestyle'] = ':'

default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
file_name_addendum = 'unconstrained'

def plot_properties(properties_folder, config_subset=None):
    property_data = {}

    property_labels = {'arithmetic_mu': 'Expected Return', 'sigma': 'Volatility',
                       'naive_md': 'Naive Maximum Drawdown', 'adjusted_md': 'Adjusted Maximum Drawdown',
                       'enb': 'Effective Number of Bets', 'div_ratio_sqrd': 'Diversification Ratio Squared',
                       'caaf_implied_epsilon': 'Implied Epsilon', 'tracking_error': 'Tracking Error',
                       'effective_rank': 'Effective Rank'
                       }
    method_labels = {'23_current_caaf': 'CAAF 1.0',
                     '23_max_div': 'CAAF 2.0',
                     '23_current_caaf_erc': 'ERC',
                     '20_current_caaf': 'CAAF 1.0',
                     '20_max_div': 'CAAF 2.0',
                     '25_current_caaf': 'CAAF 1.0',
                     '25_max_div': 'CAAF 2.0',
                     '25_current_caaf_erc': 'ERC',
                     '26_current_caaf': 'CAAF 1.0',
                     '26_max_div': 'CAAF 2.0',
                     '26_current_caaf_erc': 'ERC',
                     '27_current_caaf': 'CAAF 1.0',
                     '27_max_div': 'CAAF 2.0 Constrained',
                     '27_current_caaf_erc': 'ERC',
                     '28_current_caaf': 'CAAF 1.0',
                     '28_max_div': 'CAAF 2.0',
                     '28_current_caaf_erc': 'ERC',
                     '29_current_caaf': 'CAAF 1.0',
                     '29_max_div': 'CAAF 2.0 TE Constrained',
                     '29_current_caaf_erc': 'ERC',
                     '30_current_caaf': 'CAAF 1.0',
                     '30_max_div': 'CAAF 2.0 TE Constrained',
                     '30_current_caaf_erc': 'ERC',
                        '31_current_caaf': 'CAAF 1.0',
                        '31_max_div': 'CAAF 2.0',
                        '31_current_caaf_erc': 'ERC',
                        '32_current_caaf': 'CAAF 1.0',
                        '32_max_div': 'CAAF 2.0',
                        '32_current_caaf_erc': 'ERC',
                        '33_current_caaf': 'CAAF 1.0',
                        '33_max_div': 'CAAF 2.0 with Trigger Mechanism',
                        '33_current_caaf_erc': 'ERC',
                        '34_current_caaf': 'CAAF 1.0',
                        '34_max_div': 'CAAF 2.0',
                        '34_current_caaf_erc': 'ERC',
                        '35_current_caaf': 'CAAF 1.0',
                        '35_max_div': 'CAAF 2.0',
                        '35_current_caaf_erc': 'ERC',
                        '36_current_caaf': 'CAAF 1.0',
                        '36_max_div': 'CAAF 2.0',
                        '36_current_caaf_erc': 'ERC',
                        '37_current_caaf': 'CAAF 1.0',
                        '37_max_div': 'CAAF 2.0',
                        '37_current_caaf_erc': 'ERC',
                        '38_current_caaf': 'CAAF 1.0',
                        '38_max_div': 'CAAF 2.0',
                        '38_current_caaf_erc': 'ERC',
                        '39_current_caaf': 'CAAF 1.0',
                        '39_max_div': 'CAAF 2.0 TE Constrained',
                        '39_current_caaf_erc': 'ERC',
                        '41_current_caaf': 'CAAF 1.0',
                        '41_max_div': 'CAAF 2.0',
                        '41_current_caaf_erc': 'ERC',
                     }

    for file_name in os.listdir(properties_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(properties_folder, file_name)

            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            config_method = file_name.replace("properties_", "").replace(".csv", "")
            config_number = int(config_method.split('_')[0])

            if config_subset is None or config_number in config_subset:
                for column in df.columns:
                    if column not in property_data:
                        property_data[column] = []
                    property_data[column].append((config_method, df[column]))

    for property_name, property_list in property_data.items():
        plt.figure(figsize=(16, 8))

        # property_list = sorted(property_list, key=lambda x: 'erc' not in x[0])
        # property_list = [item for item in property_list if 'current_caaf' not in item[0]] + [item for item in property_list if 'current_caaf_erc' in item[0]]
        # property_list = [item for item in property_list if '28_current_caaf_erc' not in item[0]]

        for config_method, series in property_list:
            if config_method == '41_current_caaf_erc':
                plt.plot(series.index, series, label=method_labels[config_method], color=default_colors[5])
            else:
                plt.plot(series.index, series, label=method_labels[config_method])

        if (property_name == 'arithmetic_mu' or property_name == 'sigma' or
            property_name == 'tracking_error' or property_name == 'caaf_implied_epsilon'):
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # `1` here means values are expected to be in 0-1 range

            # plt.title(f'{property_labels[property_name]}')  # Title font size
            plt.title('')
            plt.legend()
            plt.grid(True)

        plt.xlabel('')  # X-axis label font size
        # plt.ylabel(property_labels[property_name])  # Y-axis label font size
        plt.ylabel('')
        # plt.title(f'{property_labels[property_name]}')  # Title font size
        plt.title('')
        plt.xticks()
        plt.yticks()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(f'./plots/properties/', f'{file_name_addendum}_{property_name}_with_erc.png'), format='png')
        plt.show()

    for property_name, property_list in property_data.items():
        plt.figure(figsize=(16, 8))

        property_list = [item for item in property_list if 'erc' not in item[0]]
        # property_list = [item for item in property_list if 'current_caaf' not in item[0]] + [item for item in property_list if 'current_caaf_erc' in item[0]]
        # property_list = [item for item in property_list if '28_current_caaf_erc' not in item[0]]

        for config_method, series in property_list:
            if config_method == '41_current_caaf_erc':
                plt.plot(series.index, series, label=method_labels[config_method], color=default_colors[5])
            else:
                plt.plot(series.index, series, label=method_labels[config_method])

        if (property_name == 'arithmetic_mu' or property_name == 'sigma' or
            property_name == 'tracking_error' or property_name == 'caaf_implied_epsilon'):
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # `1` here means values are expected to be in 0-1 range

            # plt.title(f'{property_labels[property_name]}')  # Title font size
            plt.title('')
            plt.legend()
            plt.grid(True)

        plt.xlabel('')  # X-axis label font size
        # plt.ylabel(property_labels[property_name])  # Y-axis label font size
        plt.ylabel('')
        # plt.title(f'{property_labels[property_name]}')  # Title font size
        plt.title('')
        plt.xticks()
        plt.yticks()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(f'./plots/properties/', f'{file_name_addendum}_{property_name}_without_erc.png'), format='png')
        plt.show()

properties_folder = './properties'
config_subset = [41]
plot_properties(properties_folder, config_subset)