import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import time
import plotly.express as px
import webbrowser
import re
import warnings
import bt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import quantstats as qs
import os

from reporting.tools.style import set_clbrm_style
set_clbrm_style(caaf_colors=True)

warnings.simplefilter(action='default', category=RuntimeWarning)

def read_configs(file_path='./Configs.xlsx'):
    configs = pd.read_excel(file_path, sheet_name='Sheet1')
    return configs.to_dict('records')
def get_config(config_number):
    return [config for config in all_configs if config['Config'] == config_number][0]

all_configs = read_configs()
num_config = 25
selected_config = get_config(num_config)

version_number = selected_config['Config']
country = selected_config['Country']
target_volatility = selected_config['Target Volatility']
additional_constraints = selected_config['Additional Constraints']
tracking_error_constraint = selected_config['Tracking Error Constraint']
tracking_error_limit = selected_config['Tracking Error Limit']

if tracking_error_constraint != 'Yes':
    tracking_error_limit = None

if country == 'US' or country == 'UK' or country == 'JP':
    equities = 'Equities'
elif np.isnan(country):
    equities = 'DM Equities'
    benchmarks = pd.read_excel(f'./data/benchmarks.xlsx', sheet_name='Sheet1')
    benchmarks.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    benchmarks['Date'] = pd.to_datetime(benchmarks['Date'], format='%Y-%m-%d')
    benchmarks.set_index('Date', inplace=True)

if additional_constraints == 'None':
    additional_constraints = None
elif additional_constraints == 'Yes':
    additional_constraints = {'alternatives_upper_bound': 0.144,
                              'em_equities_upper_bound': 0.3,
                              'hy_credit_upper_bound': 0.086,}

two_stage_target_md = "frontier_only"

# Version number 1: US Base two_stage target_md 0.27
# Version number 2: US Base two_stage target_md "varying"
# Version number 3: US Base two_stage target_md "frontier"
# Version number 4: Placeholder
# Version number 5: US Frontier Only 7%
# Version number 6: US Frontier Only 8%
# Version number 7: US Frontier Only 9%
# Version number 8: US Frontier Only 10%
# Version number 9: US Frontier Only 7% Rebalancing Trigger
# Version number 10: US Frontier Only 8% Rebalancing Trigger
# Version number 11: US Frontier Only 9% Rebalancing Trigger
# Version number 12: UK Base two_stage target_md "varying"

# Version number 13: US Frontier

################################################################################
# Color Definition
def to_percentage(x, pos):
    return '{:.1f}%'.format(x * 100)

def rgb_to_hex(rgb_tuple):
    return f"#{int(rgb_tuple[0] * 255):02x}{int(rgb_tuple[1] * 255):02x}{int(rgb_tuple[2] * 255):02x}"

default_colors = [
    (64 / 255, 75 / 255, 151 / 255),
    (154 / 255, 183 / 255, 235 / 255),
    (144 / 255, 143 / 255, 74 / 255),
    (216 / 255, 169 / 255, 23 / 255),
    (160 / 255, 84 / 255, 66 / 255),
    (189 / 255, 181 / 255, 19 / 255),
    (144 / 255, 121 / 255, 65 / 255)
]
hex_colors = [rgb_to_hex(color) for color in default_colors]

if country == 'US':
    color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Alternatives": "rgb(160, 84, 66)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)"
    }
else:
    color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "DM Equities": "rgb(64, 75, 151)",
    "EM Equities": "rgb(131, 140, 202)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Alternatives": "rgb(160, 84, 66)",
    "Gold": "rgb(216, 169, 23)"
    }

if country == 'US':
    color_palette = {
    "Equities": (64/255, 75/255, 151/255),
    "Gov Bonds": (144/255, 143/255, 74/255),
    "Alternatives": (160/255, 84/255, 66/255),
    "HY Credit": (154/255, 183/255, 235/255),
    "Gold": (216/255, 169/255, 23/255)
    }
else:
    color_palette = {
    "Equities": (64/255, 75/255, 151/255),
    "DM Equities": (64 / 255, 75 / 255, 151 / 255),
    "EM Equities": (64 / 255, 75 / 255, 151 / 255),
    "HY Credit": (154/255, 183/255, 235/255),
    "Gov Bonds": (144/255, 143/255, 74/255),
    "Alternatives": (160/255, 84/255, 66/255),
    "Gold": (216/255, 169/255, 23/255)
    }

def rgb_to_tuple(rgb_str):
    """Converts 'rgb(a, b, c)' string to a tuple of floats scaled to [0, 1]"""
    nums = re.findall(r'\d+', rgb_str)
    return tuple(int(num) / 255.0 for num in nums)

# Convert the colors in color_mapping to tuples
tuple_color_mapping = {key: rgb_to_tuple(value) for key, value in color_mapping.items()}

################################################################################
# Function Definition

def calculate_tracking_error(portfolio_returns, benchmark_returns):
    diff_returns = portfolio_returns - benchmark_returns
    tracking_error = np.sqrt(np.var(diff_returns, ddof=1))
    return tracking_error

def calculate_rolling_tracking_error(portfolio_returns, benchmark_returns, window=120):
    """
    Calculate 10-year rolling tracking error.

    Parameters:
    - portfolio_returns: Pandas Series of portfolio returns
    - benchmark_returns: Pandas Series of benchmark returns
    - window: Number of periods in rolling window, default is 120 for 10 years

    Returns:
    - Rolling Tracking Errors: Pandas Series
    """
    indices = []
    rolling_tracking_errors = []

    for i in range(len(portfolio_returns) - window + 1):
        start_idx = portfolio_returns.index[i]
        end_idx = portfolio_returns.index[i + window - 1]

        sub_portfolio = portfolio_returns.loc[start_idx:end_idx]
        sub_benchmark = benchmark_returns.loc[start_idx:end_idx]

        te = calculate_tracking_error(sub_portfolio, sub_benchmark)

        indices.append(end_idx)
        rolling_tracking_errors.append(te)

    return pd.Series(rolling_tracking_errors, index=indices)

def plot_columns(dataframe, method, country, version_number):
    dataframe['sigma'] = dataframe['sigma'].round(8)

    fig, axes = plt.subplots(nrows=len(dataframe.columns), ncols=1, figsize=(10, 12))
    for i, column in enumerate(dataframe.columns):
        ax = axes[i]
        ax.plot(dataframe.index, dataframe[column])
        ax.set_title(f"{column}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

    if method is not None:
        plt.suptitle(f"Analysis using method: {method}", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if method is not None and country is not None and version_number is not None:
        plt.savefig(f"./images/properties_{method}_{country}_{version_number}.png", dpi=300)

    plt.show()

def percentage_formatter(x, pos):
    """Format y-axis values as percentages."""
    return f"{100 * x:.0f}%"

def drop_initial_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop initial rows with duplicated values, keeping the last of the duplicates.

    Parameters:
    - df (pd.DataFrame): Input DataFrame

    Returns:
    - pd.DataFrame: DataFrame with the initial duplicated rows removed
    """
    first_non_duplicate_index = df[~df.duplicated(keep='last')].index.min()
    return df.loc[first_non_duplicate_index:]

################################################################################
# Read in Files

if country == 'US' or country == 'UK' or country == 'JP':
    rdf = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="cov")
    rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
    rdf.set_index('Date', inplace=True)
    # rdf.dropna(inplace=True)

    const_covar = rdf.cov()
    const_covar_scaled = const_covar * 12
    const_covar_scaled.to_parquet("const_covar_scaled.parquet")

    er = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="expected_gross_return")
    er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
    er.set_index('Date', inplace=True)
    # er.dropna(inplace=True)
elif np.isnan(country):
    rdf = pd.read_excel(f"./data/2024-08-31 master_file.xlsx", sheet_name="cov")
    rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
    # rdf = rdf.loc[rdf.loc[:, 'Date'] >= '1993-01-31', :]
    rdf.set_index('Date', inplace=True)
    # rdf.dropna(inplace=True)

    const_covar = rdf.cov()
    const_covar_scaled = const_covar * 12
    const_covar_scaled.to_parquet("const_covar_scaled.parquet")

    rdf = pd.read_excel(f"./data/2024-08-31 master_file.xlsx", sheet_name="cov")
    rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
    rdf = rdf.loc[rdf.loc[:, 'Date'] >= '1993-01-31', :]
    rdf.set_index('Date', inplace=True)
    # rdf.dropna(inplace=True)

    er = pd.read_excel(f"./data/2024-08-31 master_file.xlsx", sheet_name="expected_gross_return")
    er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
    er.set_index('Date', inplace=True)
    er.drop(columns=['Cash'], inplace=True)
    # er.dropna(inplace=True)

def to_percent(y, position):
    return f"{y * 100:.0f}%"

plt.figure(figsize=(15, 6))
sns.lineplot(data=er, dashes=False, palette=color_palette)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.xlabel("")
plt.ylabel("Expected Return")
plt.title("Expected Returns of Asset Classes")
plt.savefig(f'./images/expected_returns_asset_classes_{country}.png', format='png', dpi=300)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.tight_layout()
plt.show()

# Remove Expected Return for Gold until the Gold Standard
# Kept it for the chart above
er.loc[:"1973-01-31", "Gold"] = np.nan

#reference_properties = pd.read_csv(f"./data/portfolio_properties/Properties_CurrentCAAF_{source_version_number}.csv", index_col=0)
#target_md = reference_properties['adjusted_md']

if 'Cash' in er.columns:
    er.drop(columns=['Cash'], inplace=True)

if country == 'US' or country == 'UK' or country == 'JP':
    asset_class_subset = [equities, 'Gov Bonds', 'Gold',
                          'Alternatives']
    if country == 'US':
        asset_class_subset += ['HY Credit']
else:
    asset_class_subset = [equities, 'EM Equities', 'HY Credit',
                          'Gov Bonds', 'Gold', 'Alternatives']

pdf = 100*np.cumprod(1+rdf)
pdf_melted = pdf.reset_index().melt(id_vars=['Date'], value_name='Price', var_name='Asset')

# Plot using seaborn
plt.figure(figsize=(15, 6))
sns.lineplot(x='Date', y='Price', hue='Asset', data=pdf_melted, palette=color_palette)
plt.xlabel("")
plt.yscale('log')  # Log scale
plt.title('Asset Prices Over Time')
plt.savefig(f'./images/asset_prices_{country}.png', format='png', dpi=300)
plt.tight_layout()
plt.show()

selectTheseAlgo = bt.algos.SelectThese(asset_class_subset)

# algo to set the weights so each asset contributes the same amount of risk
#  with data over the last 6 months excluding yesterday
weighERCAlgo = bt.algos.WeighERC(
    lookback=pd.DateOffset(months=1000),
    covar_method='constant',
    risk_parity_method='slsqp',
    maximum_iterations=100,
    tolerance=1e-9,
    lag = pd.DateOffset(months=0)
)

# additional_constraints = {'alternatives_upper_bound': 0.144,
#                           'em_equities_upper_bound': 0.3,
#                           'hy_credit_upper_bound': 0.086,}

weighMaxDivAlgo = bt.algos.WeighMaxDiv(
    lookback=pd.DateOffset(months=1000),
    initial_weights=None,
    risk_weights=None,
    covar_method="constant",
    risk_parity_method="slsqp",
    maximum_iterations=100,
    tolerance=1e-4,
    lag=pd.DateOffset(days=0),
    bounds=(0.0, 1.0),
    additional_constraints=additional_constraints,
    mode="long_term",
    version_number=version_number,
    target_volatility=target_volatility,
)

weighCurrentCAAFAlgo = bt.algos.WeighCurrentCAAF(
    lookback=pd.DateOffset(months=1000),
    initial_weights=None,
    risk_weights=None,
    covar_method="constant",
    risk_parity_method="slsqp",
    maximum_iterations=100,
    tolerance=1e-4,
    lag=pd.DateOffset(days=0),
    bounds=(0.0, 1.0),
    additional_constraints=additional_constraints,
    mode="long_term",
    target_volatility=target_volatility +0.036,
    rdf=rdf,
    country=country
)

weighCurrentCAAFMVAlgo = bt.algos.WeighCurrentCAAFMV(
    lookback=pd.DateOffset(months=1000),
    initial_weights=None,
    risk_weights=None,
    covar_method="constant",
    risk_parity_method="slsqp",
    maximum_iterations=100,
    tolerance=1e-4,
    lag=pd.DateOffset(days=0),
    bounds=(0.0, 1.0),
    additional_constraints=additional_constraints,
    mode="long_term",
    target_volatility=target_volatility+0.004,
    rdf=rdf
)

weighCurrentCAAFERCAlgo = bt.algos.WeighCurrentCAAFERC(
    lookback=pd.DateOffset(months=1000),
    initial_weights=None,
    risk_weights=None,
    covar_method="constant",
    risk_parity_method="slsqp",
    maximum_iterations=100,
    tolerance=1e-4,
    lag=pd.DateOffset(days=0),
    bounds=(0.0, 1.0),
    additional_constraints=additional_constraints,
    mode="long_term",
    target_volatility=target_volatility+0.004,
    rdf=rdf
)

runMonthlyAlgo = bt.algos.RunMonthly(
    run_on_first_date=True,
    run_on_end_of_period=True
)
weights = pd.Series([0.4, 0.6], index=[equities, 'Gov Bonds'])
weigh4060Algo = bt.algos.WeighSpecified(**weights)

weights = pd.Series([0.6, 0.4], index=[equities, 'Gov Bonds'])
weigh6040Algo = bt.algos.WeighSpecified(**weights)

weights = pd.Series([0.5336, 0.4664], index=[equities, 'Gov Bonds'])
weighFlexAlgo = bt.algos.WeighSpecified(**weights)

weighEquallyAlgo = bt.algos.WeighEqually()

weighRandomlyAlgo = bt.algos.WeighRandomly()

weighERC = bt.algos.WeighERC(
    lookback=pd.DateOffset(months=1000),
    initial_weights=None,
    risk_weights=None,
    covar_method="constant",
    risk_parity_method="slsqp",
    maximum_iterations=100,
    tolerance=1e-4,
    lag=pd.DateOffset(days=0),
    additional_constraints=additional_constraints,
)

rebalAlgo = bt.algos.Rebalance()

# strat = bt.Strategy(
#     'ERC',
#     [
#         bt.algos.RunAfterDate('1999-01-01'),
#         selectTheseAlgo,
#         weighERCAlgo,
#         rebalAlgo
#     ]
# )

# common_start_date = '1962-04-30'
common_start_date = '1900-01-31'

tolerance = 0.2
strat_current_caaf = bt.Strategy(
    'Current CAAF',
    [
        bt.algos.ExpectedReturns('expected_returns'),
        bt.algos.ConstantCovar('const_covar'),
        bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        weighCurrentCAAFAlgo,
        # bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_current_caaf.perm["properties"] = pd.DataFrame()

strat_current_caaf_mv = bt.Strategy(
    'Current CAAF MV',
    [
        bt.algos.ExpectedReturns('expected_returns'),
        bt.algos.ConstantCovar('const_covar'),
        bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        weighCurrentCAAFMVAlgo,
        # bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_current_caaf_mv.perm["properties"] = pd.DataFrame()

strat_current_caaf_erc = bt.Strategy(
    'Current CAAF ERC',
    [
        bt.algos.ExpectedReturns('expected_returns'),
        bt.algos.ConstantCovar('const_covar'),
        bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        weighCurrentCAAFERCAlgo,
        # bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_current_caaf_erc.perm["properties"] = pd.DataFrame()

# 1973-07-31
strat_max_div = bt.Strategy(
    'Max Div',
    [
        bt.algos.ExpectedReturns('expected_returns'),
        bt.algos.ConstantCovar('const_covar'),
        bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        weighMaxDivAlgo,
        # bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_max_div.perm["properties"] = pd.DataFrame()

strat_4060 = bt.Strategy(
    '40/60',
    [
        bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        runMonthlyAlgo,
        weigh4060Algo,
        rebalAlgo
    ]
)
strat_4060.perm["properties"] = pd.DataFrame()

strat_6040 = bt.Strategy(
    '60/40',
    [
        bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        runMonthlyAlgo,
        weigh6040Algo,
        rebalAlgo
    ]
)
strat_6040.perm["properties"] = pd.DataFrame()

strat_flex = bt.Strategy(
    'Flex',
    [
        bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        runMonthlyAlgo,
        weighFlexAlgo,
        rebalAlgo
    ]
)
strat_flex.perm["properties"] = pd.DataFrame()

strat_equal = bt.Strategy(
    'Equal Weights',
    [
bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        runMonthlyAlgo,
        weighEquallyAlgo,
        rebalAlgo
    ]
)
strat_equal.perm["properties"] = pd.DataFrame()

strat_erc = bt.Strategy(
    'ERC',
    [
bt.algos.RunAfterDate(common_start_date),
        selectTheseAlgo,
        runMonthlyAlgo,
        weighERCAlgo,
        rebalAlgo
    ]
)
strat_erc.perm["properties"] = pd.DataFrame()

additional_data = {
    'expected_returns': er,
    'const_covar': const_covar,
}
backtest_current_caaf = bt.Backtest(
     strat_current_caaf,
     pdf,
     integer_positions=False,
     additional_data=additional_data,
 )
backtest_current_caaf_mv = bt.Backtest(
        strat_current_caaf_mv,
        pdf,
        integer_positions=False,
        additional_data=additional_data,
    )
backtest_current_caaf_erc = bt.Backtest(
        strat_current_caaf_erc,
        pdf,
        integer_positions=False,
        additional_data=additional_data,
    )
additional_data = {
    'expected_returns': er,
    'const_covar': const_covar,
}
backtest_max_div = bt.Backtest(
    strat_max_div,
    pdf,
    integer_positions=False,
    additional_data=additional_data
)

backtest_4060 = bt.Backtest(
    strat_4060,
    pdf,
)
backtest_6040 = bt.Backtest(
    strat_6040,
    pdf,
)
backtest_flex = bt.Backtest(
    strat_flex,
    pdf,
)
backtest_equal = bt.Backtest(
    strat_equal,
    pdf,
)
backtest_erc = bt.Backtest(
    strat_erc,
    pdf,
    additional_data={'expected_returns': er, 'const_covar': const_covar},
)

start_time = time.time()
# res_target = bt.run(backtest_current_caaf, backtest_max_div, backtest_erc, backtest_equal, backtest_6040, backtest_4060)
# res_target = bt.run(backtest_current_caaf, backtest_max_div)
# res_target = bt.run(backtest_current_caaf, backtest_max_div, backtest_4060)
# res_target = bt.run(backtest_max_div)
res_target = bt.run(backtest_current_caaf)

# res_target = bt.run(backtest_4060, backtest_6040)
# res_target = bt.run(backtest_current_caaf, backtest_max_div, backtest_current_caaf_erc,
#                     backtest_4060, backtest_6040, backtest_flex)

res_target.get_security_weights(0)

# res_target = bt.run(backtest_two_stage)
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

# Backtest Finished
####################################################################################################

####################################################################################################
# Plots & Reporting

def plot_stacked_area(df, method=None, country=None, version_number=None):
    x = pd.to_datetime(df.index).to_numpy()
    y = df.to_numpy().T

    method_labels = {'current_caaf': 'CAAF 1.0',
                     'erc': 'ERC', 'equal_weights': 'Equal Weights',
                     '6040': '60/40', '4060': '40/60',
                     'max_div': 'CAAF 2.0', 'current_caaf_mv': 'Current CAAF MV',
                     'current_caaf_erc': 'Current CAAF ERC'}

    fig, ax = plt.subplots(figsize=(18, 10))

    ax.stackplot(x, y, labels=df.columns, colors=[tuple_color_mapping[col] for col in df.columns], alpha=0.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=48))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.set_ylabel("Weight", fontsize=18)

    ax.tick_params(axis='x', rotation=45, labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)

    plt.title(f"{method_labels[method]} Weights Over Time", fontsize=20)

    plt.tight_layout()

    plt.ylabel("Weight")
    if method is not None and country is not None and version_number is not None:
        plt.savefig(f"./images/weights_{method}_{country}_{version_number}.png", dpi=300)

    plt.show()

bt_keys = list(res_target.keys())

# Stacked Area Chart for Security Weights
for method in ['Current CAAF', 'Two Stage', 'ERC', 'Equal Weights', '60/40', '40/60', 'Max Div', 'Current CAAF MV', 'Current CAAF ERC']:
    if method in bt_keys:
        if country == 'US':
            columns_to_plot = ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"] if method != '60/40' and method != '40/60' else ["Equities", "Gov Bonds"]
        elif country == 'UK' or country == 'JP':
            columns_to_plot = ["Equities", "Gov Bonds", "Alternatives", "Gold"] if method != '60/40' and method != '40/60' else ["Equities", "Gov Bonds"]
        elif country != 'UK' or country != 'US' or np.isnan(country):
            columns_to_plot = [equities, "EM Equities", "HY Credit", "Gov Bonds", "Gold", "Alternatives"] if method != '60/40' and method != '40/60' else [equities, "Gov Bonds"]
        else:
            columns_to_plot = ["Equities", "Gov Bonds", "Alternatives", "Gold"] if method != '60/40' and method != '40/60' else ["Equities", "Gov Bonds"]
        try:
            plot_stacked_area(res_target.get_security_weights(bt_keys.index(method)).loc[:, columns_to_plot], method.lower().replace(" ", "_").replace("/", "_"), country, version_number)
        except:
            print('Break')

####################################################################################################
# ENB Comparison Chart

enb_df = pd.DataFrame()
for method in ['Current CAAF', 'Max Div', 'Current CAAF MV', 'Current CAAF ERC']:
    if method in res_target.backtests:
        current_series = res_target.backtests[method].strategy.perm['properties']['enb']
        current_series.name = f'{method}'
        enb_df = pd.concat([enb_df, current_series], axis=1)

enb_df.index = pd.to_datetime(enb_df.index)

enb_df.plot(figsize=(10, 6))

plt.title('Effective Number of Bets Over Time')
plt.xlabel('Date')
plt.ylabel('Effective Number of Bets')

plt.legend(loc='best')
plt.grid(True)
plt.show()

####################################################################################################
# Properties Chart

for method in ['Current CAAF', 'Max Div', 'Current CAAF MV', 'Current CAAF ERC']:
    if method in res_target.backtests:
        current_df = res_target.backtests[method].strategy.perm['properties']

        plot_columns(res_target.backtests[method].strategy.perm['properties'], method.lower().replace(" ", "_").replace("/", "_"), country, version_number)

        print(f"Mean of sigma of {method}: {current_df['sigma'].mean()}")

        file_name = f"properties_{num_config}_{method.lower().replace(' ', '_').replace('/', '_')}.csv"
        file_path = os.path.join('./properties', file_name)
        current_df.to_csv(file_path)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(current_df.index, current_df['caaf_implied_epsilon'])
        plt.xlabel('', fontsize=16)
        plt.ylabel('CAAF Implied Epsilon', fontsize=16)

        # Set the formatter
        formatter = FuncFormatter(to_percentage)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.title(f'{method} Implied Epsilon over time', fontsize=18)
        plt.grid(True)
        plt.show()

################################################################################
# Price Plot

# For extracting prices
for method, var_name in zip(['Two Stage', 'Current CAAF', '40/60', '60/40', 'Flex', 'Max Div', 'Current CAAF MV', 'Current CAAF ERC'], ['prices_twostage', 'prices_currentcaaf', 'prices_4060', 'prices_6040', 'prices_flex', 'prices_maxdiv', 'prices_currentcaaf_mv', 'prices_currentcaaf_erc']):
    if method in res_target.prices.columns:
        exec(f"{var_name} = res_target.prices.loc[:, '{method}']")

# Initialize an empty DataFrame to store price series
combined_prices = pd.DataFrame()

# Loop through each portfolio methodology to extract and store prices
for method, var_name in zip(['Two Stage', 'Current CAAF', '40/60', '60/40', 'Flex', 'Max Div', 'Current CAAF MV', 'Current CAAF ERC'], ['prices_twostage', 'prices_currentcaaf', 'prices_4060', 'prices_6040', 'prices_flex', 'prices_maxdiv', 'prices_currentcaaf_mv', 'prices_currentcaaf_erc']):
    if method in res_target.prices.columns:
        exec(f"{var_name} = res_target.prices.loc[:, '{method}']")
        exec(f"combined_prices['{method}'] = {var_name}")

# Melt the DataFrame to long-format for Seaborn plotting
combined_prices_long = combined_prices.reset_index().melt('index', var_name='Method', value_name='Prices')

if 'Current CAAF' in bt_keys:
    combined_prices_long.loc[:, 'Method'] = combined_prices_long.loc[:, 'Method'].replace('Current CAAF', 'CAAF 1.0')
if 'Max Div' in bt_keys:
    combined_prices_long.loc[:, 'Method'] = combined_prices_long.loc[:, 'Method'].replace('Max Div', 'CAAF 2.0')

# Create figure and set size
plt.figure(figsize=(14, 6))
plt.title('Price Series Over Time', fontsize=18)
sns.lineplot(
    data=combined_prices_long,
    x='index',
    y='Prices',
    hue='Method',
    hue_order=['CAAF 1.0', 'CAAF 2.0', '40/60']  # Specify the order here
)
plt.yscale("log")
plt.xlabel("", fontsize=16)
plt.ylabel("Prices", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"./images/price_series_log_scale_{country}_{version_number}.png", dpi=300)
plt.show()

################################################################################
# Tracking Error
# Calculate tracking error if both methods in the pair are in bt_keys
if 'Current CAAF' in bt_keys and '40/60' in bt_keys:
    calculate_tracking_error(prices_currentcaaf.pct_change(), prices_4060.pct_change())

if 'Max Div' in bt_keys and '40/60' in bt_keys:
    calculate_tracking_error(prices_maxdiv.pct_change(), prices_4060.pct_change())

# Calculate and plot rolling tracking error if both methods in the pair are in bt_keys
if 'Current CAAF' in bt_keys and '40/60' in bt_keys:
    calc_track_err = calculate_rolling_tracking_error(prices_currentcaaf.pct_change(), prices_4060.pct_change())
    calc_track_err *= np.sqrt(12)
    calc_track_err.plot()
    plt.savefig(f"./images/rolling_tracking_error_current_caaf_40_60_{country}_{version_number}.png", dpi=300)
    plt.show()

# Calculate and plot rolling tracking error if both methods in the pair are in bt_keys
if 'Current CAAF' in bt_keys and '40/60' in bt_keys:
    calc_track_err = calculate_rolling_tracking_error(prices_currentcaaf.pct_change(), prices_4060.pct_change())
    calc_track_err *= np.sqrt(12)
    calc_track_err.plot()
    plt.savefig(f"./images/rolling_tracking_error_current_caaf_40_60_{country}_{version_number}.png", dpi=300)
    plt.show()

if 'Max Div' in bt_keys and '40/60' in bt_keys:
    calc_track_err = calculate_rolling_tracking_error(prices_maxdiv.pct_change(), prices_4060.pct_change())
    calc_track_err *= np.sqrt(12)
    calc_track_err.plot()
    plt.savefig(f"./images/rolling_tracking_error_max_div_40_60_{country}_{version_number}.png", dpi=300)
    plt.show()

################################################################################
# QuantStats Reports

if 'Current CAAF' in bt_keys and '40/60' in bt_keys:
    qs.reports.html(prices_currentcaaf.pct_change(), title="CAAF 1.0",
                    benchmark=prices_flex.pct_change(), benchmark_label="Flex",
                    periods_per_year=12,
                    output=f"./qs_reports/Current_CAAF_{num_config}.html",
                    download_filename=f"./qs_reports/Current_CAAF_{num_config}.html")

if 'Current CAAF MV' in bt_keys:
    qs.reports.html(prices_currentcaaf.pct_change(), "Current CAAF MV", periods_per_year=12,
                    output=f"./qs_reports/Current_CAAF_MV_{num_config}.html",
                    download_filename=f"./qs_reports/Current_CAAF_MV_{num_config}.html")

if 'Current CAAF ERC' in bt_keys:
    qs.reports.html(prices_currentcaaf.pct_change(), "Current CAAF ERC", periods_per_year=12,
                    output=f"./qs_reports/Current_CAAF_ERC_{num_config}.html",
                    download_filename=f"./qs_reports/Current_CAAF_ERC_{num_config}.html")

if 'Max Div' in bt_keys and '40/60' in bt_keys:
    qs.reports.html(prices_maxdiv.pct_change(), title="CAAF 2.0",
                    benchmark=prices_flex.pct_change(), benchmark_label="Flex",
                    periods_per_year=12,
                    output=f"./qs_reports/Maximum_Diversification_{num_config}.html",
                    download_filename=f"./qs_reports/Maximum_Diversification_{num_config}.html")

####################################################################################################
# Ratio Plots

if 'Current CAAF' in bt_keys and 'Two Stage' in bt_keys:
    ratio = res_target.prices.loc[:, "Two Stage"]/res_target.prices.loc[:, "Current CAAF"]
    ratio.plot()
    plt.savefig(f"./images/ratio_plot_current_caaf_two_stage_{country}_{version_number}.png", dpi=300)
    plt.show()

####################################################################################################
# Turnover Plots

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if 'Current CAAF' in bt_keys:
    mean_turnover = backtest_current_caaf.turnover.mean()
    print(f"Mean turnover for CAAF 1.0: {mean_turnover:.2%}")
    ax = backtest_current_caaf.turnover.plot()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Format y-axis as a percentage
    # ax.text(0.95, 0.9, f'Mean: {mean_turnover:.2%}', transform=ax.transAxes,
    #         verticalalignment='top', horizontalalignment='right',
    #         fontsize=16)
    ax.set_title("CAAF 1.0 Turnover", fontsize=16)  # Set title with font size 18
    ax.tick_params(axis='both', which='major', labelsize=14)  # Set tick label font size to 14
    ax.tick_params(axis='x', rotation=0)  # Make x-axis labels horizontal
    ax.set_ylabel("Turnover (%)", fontsize=16)  # Set y-axis label font size to 16
    plt.savefig(f"./images/turnover_current_caaf_{country}_{version_number}.png", dpi=300)
    plt.tight_layout()
    plt.show()

if 'Max Div' in bt_keys:
    mean_turnover = backtest_max_div.turnover.mean()
    print(f"Mean turnover for CAAF 2.0: {mean_turnover:.2%}")
    ax = backtest_max_div.turnover.plot()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Format y-axis as a percentage
    # ax.text(0.95, 0.9, f'Mean: {mean_turnover:.2%}', transform=ax.transAxes,
    #         verticalalignment='top', horizontalalignment='right',
    #         fontsize=16)
    ax.set_title("CAAF 2.0 Turnover", fontsize=16)  # Set title with font size 18
    ax.tick_params(axis='both', which='major', labelsize=14)  # Set tick label font size to 14
    ax.tick_params(axis='x', rotation=0)  # Make x-axis labels horizontal
    ax.set_ylabel("Turnover (%)", fontsize=16)  # Set y-axis label font size to 16
    plt.savefig(f"./images/turnover_max_div_{country}_{version_number}.png", dpi=300)
    plt.tight_layout()
    plt.show()

####################################################################################################
# Transactions Figures

for method in ['Current CAAF', 'Two Stage', 'ERC', 'Equal Weights', '60/40', '40/60']:
    if method in bt_keys:
        if country == 'US':
            columns_to_plot = ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"] if method != '60/40' and method != '40/60' else ["Equities", "Gov Bonds"]
        elif country != 'UK' or country != 'US' or np.isnan(country):
            columns_to_plot = [equities, "EM Equities", "HY Credit", "Gov Bonds", "Gold", "Alternatives"] if method != '60/40' and method != '40/60' else [equities, "Gov Bonds"]
        else:
            columns_to_plot = ["Equities", "Gov Bonds", "Alternatives", "Gold"] if method != '60/40' and method != '40/60' else ["Equities", "Gov Bonds"]
        transactions_df = res_target.get_transactions(method).reset_index()

        if country == "US":
            fig = px.line(transactions_df, x='Date', y='price', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'HY Credit': hex_colors[1], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
        else:
            fig = px.line(transactions_df, x='Date', y='price', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
        fig.update_layout(yaxis_type="log")
        fig.write_html(f"./images/transactions_{country}_{version_number}.html")

        if country == "US":
            fig = px.line(transactions_df, x='Date', y='quantity', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'HY Credit': hex_colors[1], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
        else:
            fig = px.line(transactions_df, x='Date', y='quantity', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
        fig.write_html(f"./images/quantity_{country}_{version_number}.html")


# Define a list of portfolio methodologies and corresponding DataFrame variable names
methodologies_weights = {'Current CAAF': 'current_caaf_weights', 'Two Stage': 'two_stage_weights', 'ERC': 'erc_weights', '60/40': 'weights_6040', '40/60': 'weights_4060'}
methodologies_properties = {'Current CAAF': 'current_caaf_properties', 'Two Stage': 'two_stage_properties'}

# Loop through methodologies to save security weights DataFrames
for method, var_name in methodologies_weights.items():
    if method in bt_keys:
        exec(f"{var_name} = res_target.get_security_weights(bt_keys.index('{method}'))")
        exec(f"{var_name}.to_csv(f'data/security_weights/Security_Weights_{method.replace(' ', '').replace('/', '')}_{country}_{version_number}.csv', index=True)")

# Loop through methodologies to save property DataFrames
for method, var_name in methodologies_properties.items():
    if method in bt_keys:
        exec(f"{var_name} = res_target.backtests['{method}'].strategy.perm['properties']")
        exec(f"{var_name}.to_csv(f'data/portfolio_properties/Properties_{method.replace(' ', '').replace('/', '')}_{country}_{version_number}.csv', index=True)")
