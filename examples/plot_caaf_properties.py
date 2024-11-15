import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

from reporting.tools.style import set_clbrm_style
set_clbrm_style(caaf_colors=True)

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
                     '26_current_caaf_erc': 'ERC'}

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
        plt.figure(figsize=(12, 8))

        property_list = sorted(property_list, key=lambda x: 'erc' not in x[0])

        for config_method, series in property_list:
            if config_method == '26_current_caaf_erc':
                plt.plot(series.index, series, label=method_labels[config_method], color=default_colors[5])
            else:
                plt.plot(series.index, series, label=method_labels[config_method])

        if (property_name == 'arithmetic_mu' or property_name == 'sigma' or
            property_name == 'tracking_error' or property_name == 'caaf_implied_epsilon'):
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # `1` here means values are expected to be in 0-1 range

            plt.title(f'Comparison of {property_labels[property_name]} across Configurations and Methods', fontsize=18)  # Title font size
            plt.legend(fontsize=16)
            plt.grid(True)

        plt.xlabel('', fontsize=16)  # X-axis label font size
        plt.ylabel(property_labels[property_name], fontsize=16)  # Y-axis label font size
        plt.title(f'Comparison of {property_labels[property_name]} across Configurations and Methods', fontsize=18)  # Title font size
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(f'./plots/properties/', f'{file_name_addendum}_{property_name}_with_erc.png'), format='png', dpi=300)
        plt.show()

    for property_name, property_list in property_data.items():
        plt.figure(figsize=(12, 8))

        property_list = [item for item in property_list if 'erc' not in item[0]]

        for config_method, series in property_list:
            if config_method == '26_current_caaf_erc':
                plt.plot(series.index, series, label=method_labels[config_method], color=default_colors[5])
            else:
                plt.plot(series.index, series, label=method_labels[config_method])

        if (property_name == 'arithmetic_mu' or property_name == 'sigma' or
            property_name == 'tracking_error' or property_name == 'caaf_implied_epsilon'):
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))  # `1` here means values are expected to be in 0-1 range

            plt.title(f'Comparison of {property_labels[property_name]} across Configurations and Methods', fontsize=18)  # Title font size
            plt.legend(fontsize=16)
            plt.grid(True)

        plt.xlabel('', fontsize=16)  # X-axis label font size
        plt.ylabel(property_labels[property_name], fontsize=16)  # Y-axis label font size
        plt.title(f'Comparison of {property_labels[property_name]} across Configurations and Methods', fontsize=18)  # Title font size
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(f'./plots/properties/', f'{file_name_addendum}_{property_name}_without_erc.png'), format='png', dpi=300)
        plt.show()

properties_folder = './properties'
config_subset = [26]
plot_properties(properties_folder, config_subset)
