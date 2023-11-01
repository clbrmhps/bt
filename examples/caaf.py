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

from reporting.tools.style import set_clbrm_style
set_clbrm_style(caaf_colors=True)

warnings.simplefilter(action='default', category=RuntimeWarning)

version_number = 2
source_version_number = 2
country = 'US'
two_stage_target_md = "varying"

# Version number 1: US Base two_stage target_md 0.27
# Version number 2: US Base two_stage target_md "varying"
# Version number 3: US

################################################################################
# Color Definition

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

color_mapping = {
    "Equities": "rgb(64, 75, 151)",
    "Gov Bonds": "rgb(144, 143, 74)",
    "Alternatives": "rgb(160, 84, 66)",
    "HY Credit": "rgb(154, 183, 235)",
    "Gold": "rgb(216, 169, 23)"
}

color_palette = {
    "Equities": (64/255, 75/255, 151/255),
    "Gov Bonds": (144/255, 143/255, 74/255),
    "Alternatives": (160/255, 84/255, 66/255),
    "HY Credit": (154/255, 183/255, 235/255),
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
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12))
    for i, column in enumerate(dataframe.columns):
        ax = axes[i]
        ax.plot(dataframe.index, dataframe[column])
        ax.set_title(f"{column}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

    plt.tight_layout()
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

rdf = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf.set_index('Date', inplace=True)
# rdf.dropna(inplace=True)

const_covar = rdf.cov()

er = pd.read_excel(f"./data/2023-10-26 master_file_{country}.xlsx", sheet_name="expected_gross_return")
er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
er.set_index('Date', inplace=True)
# er.dropna(inplace=True)

def to_percent(y, position):
    return f"{y * 100:.0f}%"

plt.figure(figsize=(15, 6))
sns.lineplot(data=er, dashes=False, palette=color_palette)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.xlabel("Date")
plt.ylabel("Expected Return")
plt.title("Expected Returns of Asset Classes")
plt.savefig(f'./images/expected_returns_asset_classes_{country}.png', format='png', dpi=300)
plt.show()

# Remove Expected Return for Gold until the Gold Standard
# Kept it for the chart above
er.loc[:"1973-01-31", "Gold"] = np.nan

reference_properties = pd.read_csv(f"./data/portfolio_properties/Properties_CurrentCAAF_{source_version_number}.csv", index_col=0)
target_md = reference_properties['adjusted_md']
try:
    efficient_frontier_two_stage = pd.read_pickle(f"./data/efficient_frontier_two_stage_{version_number}.pkl")
    efficient_frontier_two_stage.set_index("Date", inplace=True)
    efficient_frontier_two_stage.sort_index(inplace=True)
except FileNotFoundError:
    pass
try:
    efficient_frontier_current_caaf = pd.read_pickle(f"./data/efficient_frontier_current_caaf_{version_number}.pkl")
    efficient_frontier_current_caaf.set_index("Date", inplace=True)
    efficient_frontier_current_caaf.sort_index(inplace=True)
except FileNotFoundError:
    pass

if 'Cash' in er.columns:
    er.drop(columns=['Cash'], inplace=True)

asset_class_subset = ['Equities', 'HY Credit', 'Gov Bonds', 'Gold',
                      'Alternatives']

pdf = 100*np.cumprod(1+rdf)
# Melt the DataFrame to a long format which works better with seaborn
pdf_melted = pdf.reset_index().melt(id_vars=['Date'], value_name='Price', var_name='Asset')

# Plot using seaborn
plt.figure(figsize=(15, 6))
sns.lineplot(x='Date', y='Price', hue='Asset', data=pdf_melted, palette=color_palette)
plt.yscale('log')  # Log scale
plt.title('Asset Prices Over Time')
plt.savefig(f'./images/asset_prices_{country}.png', format='png', dpi=300)
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

additional_constraints = {'alternatives_upper_bound': 0.144,
                          'em_equities_upper_bound': 0.3,
                          'hy_credit_upper_bound': 0.086,}
additional_constraints = None

weighTwoStageAlgo = bt.algos.WeighTwoStage(
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
    return_factor=0.9,
    target_md=two_stage_target_md
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
    target_md=0.4
)

runMonthlyAlgo = bt.algos.RunMonthly(
    run_on_first_date=True,
    run_on_end_of_period=True
)
weights = pd.Series([0.4, 0.6], index=['Equities', 'Gov Bonds'])
weigh4060Algo = bt.algos.WeighSpecified(**weights)

weights = pd.Series([0.6, 0.4], index=['Equities', 'Gov Bonds'])
weigh6040Algo = bt.algos.WeighSpecified(**weights)

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

tolerance = 0.2
strat_current_caaf = bt.Strategy("Current CAAF",
    [
        bt.algos.ExpectedReturns('expected_returns'),
        bt.algos.ConstantCovar('const_covar'),
        bt.algos.RunAfterDate('1875-01-31'),
        selectTheseAlgo,
        weighCurrentCAAFAlgo,
        # bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_current_caaf.perm["properties"] = pd.DataFrame()

strat_two_stage = bt.Strategy(
    'Two Stage',
    [
        bt.algos.ExpectedReturns('expected_returns'),
        bt.algos.ConstantCovar('const_covar'),
        bt.algos.RunAfterDate('1875-01-31'),
        selectTheseAlgo,
        weighTwoStageAlgo,
        # bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_two_stage.perm["properties"] = pd.DataFrame()

strat_4060 = bt.Strategy(
    '40/60',
    [
        bt.algos.RunAfterDate('1875-01-31'),
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
        bt.algos.RunAfterDate('1875-01-31'),
        selectTheseAlgo,
        runMonthlyAlgo,
        weigh6040Algo,
        rebalAlgo
    ]
)
strat_6040.perm["properties"] = pd.DataFrame()

strat_equal = bt.Strategy(
    'Equal Weights',
    [
bt.algos.RunAfterDate('1875-01-31'),
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
bt.algos.RunAfterDate('1875-01-31'),
        selectTheseAlgo,
        runMonthlyAlgo,
        weighERCAlgo,
        rebalAlgo
    ]
)
strat_erc.perm["properties"] = pd.DataFrame()

backtest_current_caaf = bt.Backtest(
     strat_current_caaf,
     pdf,
     integer_positions=False,
     additional_data={'expected_returns': er , 'const_covar': const_covar},
 )

additional_data = {
    'expected_returns': er,
    'const_covar': const_covar,
    'target_md_var': target_md
}

if two_stage_target_md == "frontier":
    additional_data['efficient_frontier'] = efficient_frontier_two_stage

backtest_two_stage = bt.Backtest(
    strat_two_stage,
    pdf,
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
res_target = bt.run(backtest_current_caaf, backtest_two_stage, backtest_erc, backtest_equal, backtest_6040, backtest_4060)
# res_target = bt.run(backtest_current_caaf, backtest_two_stage)
# res_target = bt.run(backtest_current_caaf)

# res_target = bt.run(backtest_two_stage)
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

def plot_stacked_area(df, method=None, country=None, version_number=None):
    # Prepare data
    x = pd.to_datetime(df.index).to_numpy()
    y = df.to_numpy().T

    # Create plot and axes
    fig, ax = plt.subplots(figsize=(18, 10))

    # Generate stackplot with colors from the tuple_color_mapping
    ax.stackplot(x, y, labels=df.columns, colors=[tuple_color_mapping[col] for col in df.columns], alpha=0.6)

    # Configure x-axis for date format and ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=48))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.tick_params(axis='x', rotation=90)

    # Format the y-axis as percentages
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax.set_ylim(0, 1)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to prevent clipping
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if method is not None and country is not None and version_number is not None:
        plt.savefig(f"./images/weights_{method}_{country}_{version_number}.png", dpi=300)
    # Show plot
    plt.show()

bt_keys = list(res_target.keys())

# For plotting stacked areas based on get_security_weights
for method in ['Current CAAF', 'Two Stage', 'ERC', 'Equal Weights', '60/40', '40/60']:
    if method in bt_keys:
        columns_to_plot = ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"] if method != '60/40' and method != '40/60' else ["Equities", "Gov Bonds"]
        plot_stacked_area(res_target.get_security_weights(bt_keys.index(method)).loc[:, columns_to_plot], method.lower().replace(" ", "_").replace("/", "_"), country, version_number)

# For plotting columns from res_target.backtests
for method in ['Current CAAF', 'Two Stage']:
    if method in res_target.backtests:
        plot_columns(res_target.backtests[method].strategy.perm['properties'], method.lower().replace(" ", "_").replace("/", "_"), country, version_number)

################################################################################
# Price Plot

# For extracting prices
for method, var_name in zip(['Two Stage', 'Current CAAF', '40/60', '60/40'], ['prices_twostage', 'prices_currentcaaf', 'prices_4060', 'prices_6040']):
    if method in res_target.prices.columns:
        exec(f"{var_name} = res_target.prices.loc[:, '{method}']")

# Initialize an empty DataFrame to store price series
combined_prices = pd.DataFrame()

# Loop through each portfolio methodology to extract and store prices
for method, var_name in zip(['Two Stage', 'Current CAAF', '40/60', '60/40'], ['prices_twostage', 'prices_currentcaaf', 'prices_4060', 'prices_6040']):
    if method in res_target.prices.columns:
        exec(f"{var_name} = res_target.prices.loc[:, '{method}']")
        exec(f"combined_prices['{method}'] = {var_name}")

# Melt the DataFrame to long-format for Seaborn plotting
combined_prices_long = combined_prices.reset_index().melt('index', var_name='Method', value_name='Prices')

# Create figure and set size
plt.figure(figsize=(14, 6))
sns.lineplot(data=combined_prices_long, x='index', y='Prices', hue='Method')
plt.yscale("log")
plt.xlabel("Date")
plt.savefig(f"./images/price_series_log_scale_{country}_{version_number}.png", dpi=300)
plt.show()

################################################################################

# Calculate tracking error if both methods in the pair are in bt_keys
if 'Current CAAF' in bt_keys and '40/60' in bt_keys:
    calculate_tracking_error(prices_currentcaaf.pct_change(), prices_4060.pct_change())

if 'Two Stage' in bt_keys and '40/60' in bt_keys:
    calculate_tracking_error(prices_twostage.pct_change(), prices_4060.pct_change())

# Calculate and plot rolling tracking error if both methods in the pair are in bt_keys
if 'Current CAAF' in bt_keys and '40/60' in bt_keys:
    calc_track_err = calculate_rolling_tracking_error(prices_currentcaaf.pct_change(), prices_4060.pct_change())
    calc_track_err *= np.sqrt(12)
    calc_track_err.plot()
    plt.savefig(f"./images/rolling_tracking_error_current_caaf_40_60_{country}_{version_number}.png", dpi=300)
    plt.show()

if 'Current CAAF' in bt_keys and '60/40' in bt_keys:
    calc_track_err = calculate_rolling_tracking_error(prices_currentcaaf.pct_change(), prices_6040.pct_change())
    calc_track_err *= np.sqrt(12)
    calc_track_err.plot()
    plt.savefig(f"./images/rolling_tracking_error_current_caaf_60_40_{country}_{version_number}.png", dpi=300)
    plt.show()

if 'Current CAAF' in bt_keys and 'Two Stage' in bt_keys:
    ratio = res_target.prices.loc[:, "Two Stage"]/res_target.prices.loc[:, "Current CAAF"]
    ratio.plot()
    plt.savefig(f"./images/ratio_plot_current_caaf_two_stage_{country}_{version_number}.png", dpi=300)
    plt.show()

for method in ['Current CAAF', 'Two Stage', 'ERC', 'Equal Weights', '60/40', '40/60']:
    if method in bt_keys:
        columns_to_plot = ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"] if method != '60/40' and method != '40/60' else ["Equities", "Gov Bonds"]
        transactions_df = res_target.get_transactions(method).reset_index()

        fig = px.line(transactions_df, x='Date', y='price', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'HY Credit': hex_colors[1], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
        fig.update_layout(yaxis_type="log")
        fig.write_html(f"./images/transactions_{country}_{version_number}.html")

        fig = px.line(transactions_df, x='Date', y='quantity', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'HY Credit': hex_colors[1], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
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
