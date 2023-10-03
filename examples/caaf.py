import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import time

import ffn
import bt

from reporting.tools.style import set_clbrm_style

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

rdf = pd.read_excel("./data/2023-09-25 master_file_US.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf.set_index('Date', inplace=True)
# rdf.dropna(inplace=True)

const_covar = rdf.cov()

er = pd.read_excel("./data/2023-09-25 master_file_US.xlsx", sheet_name="expected_gross_return")
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
    return_factor=0.8,
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
        bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
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
        bt.algos.Or([bt.algos.RunOnce(), bt.algos.RunIfOutOfTriggerThreshold(tolerance)]),
        rebalAlgo
    ]
)
strat_two_stage.perm["properties"] = pd.DataFrame()

backtest_current_caaf = bt.Backtest(
     strat_current_caaf,
     pdf,
     additional_data={'expected_returns': er , 'const_covar': const_covar},
     integer_positions=False
 )
backtest_two_stage = bt.Backtest(
    strat_two_stage,
    pdf,
    additional_data={'expected_returns': er, 'const_covar': const_covar},
    integer_positions=False
)

start_time = time.time()
res_target = bt.run(backtest_current_caaf, backtest_two_stage)
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

    # Retrieve default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Skip the second color
    custom_colors = [color for i, color in enumerate(default_colors) if i != 1]
    custom_colors[4], custom_colors[3] = custom_colors[3], custom_colors[4]

    # Create plot and axes
    fig, ax = plt.subplots(figsize=(15, 10))

    # Generate stackplot with custom colors
    ax.stackplot(x, y, labels=df.columns, colors=custom_colors, alpha=0.6)

    # Get handles and labels for legend
    handles, labels = ax.get_legend_handles_labels()

    # Modify order for both labels and colors
    custom_order = [0, 1, 2, 4, 3]  # Replace this with the desired order
    ordered_handles = [handles[idx] for idx in custom_order]
    ordered_labels = [labels[idx] for idx in custom_order]
    ordered_colors = [custom_colors[idx] for idx in custom_order]

    # Apply the new color order to the stackplot
    ax.clear()  # Clear the current plot
    ax.stackplot(x, y[custom_order, :], labels=ordered_labels, colors=ordered_colors, alpha=0.6)

    plt.ylim(0, 1)
    # Configure x-axis for date format and ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=48))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.tick_params(axis='x', rotation=90)

    # Format the y-axis as percentages
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Add ordered legend
    ax.legend(ordered_handles, ordered_labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to prevent clipping
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Show plot
    plt.show()

plot_stacked_area(res_target.get_security_weights(0).loc[:, ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"]])
plot_stacked_area(res_target.get_security_weights(1).loc[:, ["Equities", "HY Credit", "Gov Bonds", "Alternatives", "Gold"]])

plot_columns(res_target.backtests['Current CAAF'].strategy.perm['properties'])
plot_columns(res_target.backtests['TwoStage'].strategy.perm['properties'])

ratio = res_target.prices.loc[:, "TwoStage"]/res_target.prices.loc[:, "Current CAAF"]
ratio.plot()
plt.show()

res_target.get_transactions().plot()
plt.show()

drop_initial_duplicates(res_target.backtest_list[0].strategy.outlays)