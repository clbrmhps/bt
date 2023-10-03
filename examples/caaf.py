import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import time
import plotly.express as px
import webbrowser
import re

import ffn
import bt

from reporting.tools.style import set_clbrm_style

version_number = 1

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

def rgb_to_tuple(rgb_str):
    """Converts 'rgb(a, b, c)' string to a tuple of floats scaled to [0, 1]"""
    nums = re.findall(r'\d+', rgb_str)
    return tuple(int(num) / 255.0 for num in nums)

# Convert the colors in color_mapping to tuples
tuple_color_mapping = {key: rgb_to_tuple(value) for key, value in color_mapping.items()}

def plot_columns(dataframe):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
    for i, column in enumerate(dataframe.columns):
        ax = axes[i]
        ax.plot(dataframe.index, dataframe[column])
        ax.set_title(f"{column}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

    plt.tight_layout()
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

set_clbrm_style(caaf_colors=True)

import warnings
warnings.simplefilter(action='default', category=RuntimeWarning)

rdf = pd.read_excel("./data/2023-10-03 master_file_US.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf.set_index('Date', inplace=True)
# rdf.dropna(inplace=True)

const_covar = rdf.cov()

er = pd.read_excel("./data/2023-10-03 master_file_US.xlsx", sheet_name="expected_gross_return")
er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
er.set_index('Date', inplace=True)
# er.dropna(inplace=True)

if 'Cash' in er.columns:
    er.drop(columns=['Cash'], inplace=True)

asset_class_subset = ['Equities', 'HY Credit', 'Gov Bonds', 'Gold',
                      'Alternatives']

pdf = 100*np.cumprod(1+rdf)
pdf.plot()
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
    return_factor=0.7,
    target_md=0.275
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
    target_md=0.35
)

runMonthlyAlgo = bt.algos.RunMonthly(
    run_on_first_date=True,
    run_on_end_of_period=True
)

weights = pd.Series([0.4, 0.6], index=['Equities', 'Gov Bonds'])
weigh4060Algo = bt.algos.WeighSpecified(**weights)

weights = pd.Series([0.6, 0.4], index=['Equities', 'Gov Bonds'])
weigh6040Algo = bt.algos.WeighSpecified(**weights)

weightEquallyAlgo = bt.algos.WeighEqually()

weighRandomlyAlgo = bt.algos.WeighRandomly()

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
        bt.algos.RunAfterDate('1871-02-28'),
        selectTheseAlgo,
        weighCurrentCAAFAlgo,
        # bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_current_caaf.perm["properties"] = pd.DataFrame()

strat_two_stage = bt.Strategy(
    'TwoStage',
    [
        bt.algos.ExpectedReturns('expected_returns'),
        bt.algos.ConstantCovar('const_covar'),
        bt.algos.RunAfterDate('1871-02-28'),
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
        bt.algos.RunAfterDate('1871-02-28'),
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
        bt.algos.RunAfterDate('1871-02-28'),
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
bt.algos.RunAfterDate('1871-02-28'),
        selectTheseAlgo,
        runMonthlyAlgo,
        weightEquallyAlgo,
        rebalAlgo
    ]
)
strat_equal.perm["properties"] = pd.DataFrame()

backtest_current_caaf = bt.Backtest(
     strat_current_caaf,
     pdf,
     additional_data={'expected_returns': er , 'const_covar': const_covar},
 )
backtest_two_stage = bt.Backtest(
    strat_two_stage,
    pdf,
    additional_data={'expected_returns': er, 'const_covar': const_covar},
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

start_time = time.time()
res_target = bt.run(backtest_two_stage)

# res_target = bt.run(backtest_two_stage)
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

res_target.display()

res_target.plot_histogram()
plt.show()

res_target.plot()
plt.show()

def plot_stacked_area(df):
    # Prepare data
    x = pd.to_datetime(df.index).to_numpy()
    y = df.to_numpy().T

    # Create plot and axes
    fig, ax = plt.subplots(figsize=(15, 10))

    # Generate stackplot with colors from the tuple_color_mapping
    ax.stackplot(x, y, labels=df.columns, colors=[tuple_color_mapping[col] for col in df.columns], alpha=0.6)

    # Configure x-axis for date format and ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=48))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.tick_params(axis='x', rotation=90)

    # Format the y-axis as percentages
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to prevent clipping
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Show plot
    plt.show()

plot_stacked_area(res_target.get_security_weights(0).loc[:, ["Equities", "Gov Bonds"]])
plot_stacked_area(res_target.get_security_weights(1).loc[:, ["Equities", "Gov Bonds"]])
plot_stacked_area(res_target.get_security_weights(2).loc[:, ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"]])
plot_stacked_area(res_target.get_security_weights(3).loc[:, ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"]])

plot_columns(res_target.backtests['Current CAAF'].strategy.perm['properties'])
plot_columns(res_target.backtests['TwoStage'].strategy.perm['properties'])

ratio = res_target.prices.loc[:, "TwoStage"]/res_target.prices.loc[:, "Current CAAF"]
ratio.plot()
plt.show()

transactions_df = res_target.get_transactions().reset_index()

fig = px.line(transactions_df, x='Date', y='price', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'HY Credit': hex_colors[1], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
fig.write_html("plot.html")
webbrowser.open("plot.html")

fig = px.line(transactions_df, x='Date', y='quantity', color='Security',
              color_discrete_map={'Equities': hex_colors[0], 'HY Credit': hex_colors[1], 'Gov Bonds': hex_colors[2], 'Gold': hex_colors[3], 'Alternatives': hex_colors[4]})
fig.write_html("plot.html")
webbrowser.open("plot.html")

res_target.get_transactions().plot()
plt.xticks(rotation=45)
plt.show()

# Assuming res_target.get_security_weights(0) and res_target.get_security_weights(1) return DataFrames
df1 = res_target.get_security_weights(0)
df2 = res_target.get_security_weights(1)

# Similarly for res_target.backtests['Current CAAF'].strategy.perm['properties'] and res_target.backtests['TwoStage'].strategy.perm['properties']
df3 = res_target.backtests['Current CAAF'].strategy.perm['properties']
df4 = res_target.backtests['TwoStage'].strategy.perm['properties']

# Save to the data folder with a timestamp
df1.to_csv(f'data/security_weights/Security_Weights_CurrentCAAF_{version_number}.csv', index=True)
df2.to_csv(f'data/security_weights/Security_Weights_TwoStage_{version_number}.csv', index=True)
df3.to_csv(f'data/portfolio_properties/Properties_CurrentCAAF_{version_number}.csv', index=True)
df4.to_csv(f'data/portfolio_properties/Properties_TwoStage_{version_number}.csv', index=True)

drop_initial_duplicates(res_target.backtest_list[0].strategy.outlays)