import warnings
import time
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import ffn
import bt
from reporting.tools.style import set_clbrm_style

from ffn.core import calc_two_stage_weights

from plotly.subplots import make_subplots

def rgb_to_hex(rgb_tuple):
    return f"#{int(rgb_tuple[0] * 255):02x}{int(rgb_tuple[1] * 255):02x}{int(rgb_tuple[2] * 255):02x}"

def generate_multiplier(n):
    # Convert the number to a string and find the position of the decimal point
    n_str = str(n)
    decimal_pos = n_str.find('.')

    # If there's no decimal, the number is an integer, return 5
    if decimal_pos == -1:
        return 5

    # Find the highest non-zero decimal place
    d = 0
    for digit in n_str[decimal_pos+1:]:
        if digit != '0':
            break
        d += 1

    # Create a number with a "1" at the highest non-zero decimal place
    new_number = 1 / (10 ** (d + 1))

    # Multiply by 1
    new_number *= 5

    return new_number

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

def percentage_formatter(x, pos):
    """Format y-axis values as percentages."""
    return f"{100 * x:.0f}%"


# Define a function to filter each group (date)
def filter_rows(group):
    max_arithmetic_mu = group['arithmetic_mu'].max() - 0.001
    sigma_at_max_mu = group.loc[group['arithmetic_mu'].idxmax(), 'sigma']

    return group[(group['arithmetic_mu'] >= max_arithmetic_mu) | (group['sigma'] <= sigma_at_max_mu)]


set_clbrm_style(caaf_colors=True)

import warnings
warnings.simplefilter(action='default', category=RuntimeWarning)

import pandas as pd

# The timestamp you used while saving
version_number = 3
country = "US"

# Read the DataFrames from the saved CSV files
weight_caaf = pd.read_csv(f'data/security_weights/Security_Weights_CurrentCAAF_{country}_{version_number}.csv', index_col=0)
weight_twostage = pd.read_csv(f'data/security_weights/Security_Weights_TwoStage_{country}_{version_number}.csv', index_col=0)
properties_caaf = pd.read_csv(f'data/portfolio_properties/Properties_CurrentCAAF_{country}_{version_number}.csv', index_col=0)
properties_twostage = pd.read_csv(f'data/portfolio_properties/Properties_TwoStage_{country}_{version_number}.csv', index_col=0)

efficient_frontier_current_caaf = pd.read_pickle(f"./data/efficient_frontier_current_caaf_{version_number}.pkl")
efficient_frontier_two_stage = pd.read_pickle(f"./data/efficient_frontier_two_stage_{version_number}.pkl")

efficient_frontier_current_caaf = efficient_frontier_current_caaf.groupby('Date').apply(filter_rows).reset_index(drop=True)
efficient_frontier_two_stage = efficient_frontier_two_stage.groupby('Date').apply(filter_rows).reset_index(drop=True)

# Original filter function for two-stage
weight_caaf.index.name = 'Date'
weight_twostage.index.name = 'Date'
properties_caaf.index.name = 'Date'
properties_twostage.index.name = 'Date'

weight_caaf.index = pd.to_datetime(weight_caaf.index)
weight_twostage.index = pd.to_datetime(weight_twostage.index)
properties_caaf.index = pd.to_datetime(properties_caaf.index)
properties_twostage.index = pd.to_datetime(properties_twostage.index)

weight_caaf = weight_caaf.iloc[1:]
weight_twostage = weight_twostage.iloc[1:]

rdf = pd.read_excel("./data/2023-10-26 master_file_US.xlsx", sheet_name="cov")
rdf['Date'] = pd.to_datetime(rdf['Date'], format='%d/%m/%Y')
rdf.set_index('Date', inplace=True)
# rdf.dropna(inplace=True)

const_covar = rdf.cov()
const_covar *= 12

er = pd.read_excel("./data/2023-10-26 master_file_US.xlsx", sheet_name="expected_gross_return")
er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
er.set_index('Date', inplace=True)

def generate_weights(minimum, maximum, increment, target):
    if (minimum.shape != maximum.shape) or (np.ndim(minimum) != 1):
        raise Exception('minimum and maximum must be of dimension 1 and of same length.')
    if np.any(minimum > maximum):
        raise Exception('minima cannot be larger than maxima.')
    if not np.all(np.isclose(np.mod(minimum / increment, 1), 0)) or \
            not np.all(np.isclose(np.mod(maximum / increment, 1), 0)):
        raise Exception('minima and maxima must be divisible by increment.')
    minimum = (minimum / increment).astype(int)
    maximum = (maximum / increment).astype(int)
    target = int(target / increment)
    # Loop
    final_list = list()
    for j in range(minimum[0], maximum[0] + 1):
        final_list.append((j,))
    for i in range(1, len(minimum)):
        temp = list()
        for current in final_list:
            for j in range(max(target - sum(current) - sum(maximum[i + 1:]), minimum[i]),
                           min(target - sum(current) - sum(minimum[i + 1:]), maximum[i]) + 1):
                temp.append(current + (j,))
        final_list = temp
    return np.array(final_list) * increment

# Make it so that er_aligned has exactly the same missingness pattern as rdf
er_aligned = er.reindex(rdf.index)
er_aligned.loc[er_aligned.index.difference(rdf.index)] = np.nan
for col in er.columns:
    er_aligned.loc[rdf[col].isna(), col] = np.nan

# Initialize variables to hold start point, list of results, and list of asset blocks
start_point = 0
block_expected_returns = []
date_indexes = []
block_stdevs = []
block_weights = []

# Iterate through the DataFrame to identify blocks and multiply
for i in range(1, len(er_aligned)):
    # Identify the dimensions of non-NA values for the current and previous rows
    curr_dims = er_aligned.iloc[i].dropna().shape[0]
    prev_dims = er_aligned.iloc[i - 1].dropna().shape[0]

    # Check if the dimensions change, indicating the start of a new block
    if curr_dims != prev_dims:
        # Extract the block and drop NA values along columns
        block = er_aligned.iloc[start_point:i].dropna(axis=1)

        # Record the assets in this block
        assets_in_block = block.columns.tolist()

        # Generate the weight matrix w for this block
        min_vals = np.zeros(block.shape[1])
        max_vals = np.ones(block.shape[1])
        w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)

        aligned_covar = const_covar.loc[assets_in_block, assets_in_block]
        block_arithmetic_return = block.values + np.diag(aligned_covar.to_numpy())/2

        # Perform the matrix multiplication and append to the result list
        block_result = block_arithmetic_return.dot(w.transpose())

        for j, date_index in enumerate(block.index):
            if j == 0:
                continue
            date_indexes.append(date_index)
            block_weights.append(w)
            block_expected_returns.append(block_result[j])

        # Generate w for the block (reuse your generate_weights function)
        # min_vals = np.zeros(len(assets_in_block))
        # max_vals = np.ones(len(assets_in_block))
        # w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)

        # Calculate the standard deviation for each weight vector in w
        stdevs = []
        for w_row in w:
            stdev = np.sqrt(w_row @ aligned_covar @ w_row.T)  # @ is matrix multiplication
            stdevs.append(stdev)

        for j, date_index in enumerate(block.index):
            block_stdevs.append(stdevs)

        # Update the start_point for the next block
        start_point = i

# Handle the last block after exiting the loop
block = er_aligned.iloc[start_point:].dropna(axis=1)

# Record the assets in this block
assets_in_block = block.columns.tolist()

min_vals = np.zeros(block.shape[1])
max_vals = np.ones(block.shape[1])
w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)

aligned_covar = const_covar.loc[assets_in_block, assets_in_block]
block_arithmetic_return = block.values + np.diag(aligned_covar.to_numpy()) / 2

block_result = block_arithmetic_return.dot(w.transpose())
for j, date_index in enumerate(block.index):
    date_indexes.append(date_index)
    block_expected_returns.append(block_result[j])
    block_weights.append(w)

# Generate w for the block (reuse your generate_weights function)
# min_vals = np.zeros(len(assets_in_block))
# max_vals = np.ones(len(assets_in_block))
# w = generate_weights(minimum=min_vals, maximum=max_vals, increment=generate_multiplier(0.05**(5/len(min_vals))), target=1)

# Calculate the standard deviation for each weight vector in w
stdevs = []
for w_row in w:
    stdev = np.sqrt(w_row @ aligned_covar @ w_row.T)  # @ is matrix multiplication
    stdevs.append(stdev)

for j, date_index in enumerate(block.index):
    block_stdevs.append(stdevs)

# import matplotlib.pyplot as plt
#
# # Loop through each date
# for i, (returns_matrix, stdevs) in enumerate(zip(block_expected_returns, block_stdevs)):
#
#     # Loop through the 348 different asset sets for a single date
#     for j in range(returns_matrix.shape[0]):
#         plt.figure(figsize=(10, 6))
#
#         # Grab the jth set of expected returns for this date
#         returns = returns_matrix[j, :]
#
#         # Plot the efficient frontier for this asset set and date
#         plt.scatter(stdevs, returns, c='blue', label='Efficient Frontier')
#
#         # Optional: add labels, titles, etc.
#         plt.xlabel('Standard Deviation')
#         plt.ylabel('Expected Return')
#         plt.title(f'Efficient Frontier for Date Index {i}, Asset Set {j + 1}')
#         plt.grid(True)
#         plt.legend(loc='upper left')
#
#         # Show plot
#         plt.show()

max_stdev = max(list(chain.from_iterable(block_stdevs)))
min_stdev = min(list(chain.from_iterable(block_stdevs)))

max_expected_return = max(np.array(list(chain.from_iterable(block_expected_returns))))
min_expected_return = min(np.array(list(chain.from_iterable(block_expected_returns))))

max_expected_return = 0.2
min_expected_return = 0

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objects as go  # For adding additional plot elements

# Initialize the Dash app
app = dash.Dash(__name__)
# App layout
# app.layout = html.Div([
#     dcc.Graph(id='graph'),
#     dcc.Slider(
#         id='date-slider',
#         min=0,
#         max=len(date_indexes) - 1,
#         value=0,
#         marks={str(i): {'label': str(date_indexes[i].year) if hasattr(date_indexes[i], 'year') else str(date_indexes[i])[:4], 'style': {'writing-mode': 'vertical-rl'}} for i in range(0, len(date_indexes), 12)},  # Rotate labels and show only year
#         step=1
#     )
# ])

# App layout
app.layout = html.Div([
    dcc.Graph(id='graph-efficient-frontier'),
    # html.Div([
    # dcc.Graph(id='graph-enb', style={'width': '100%', 'display': 'inline-block'}),
    # dcc.Graph(id='graph-adjusted_md', style={'width': '100%', 'display': 'inline-block'})
    # ], style={'display': 'flex'}),
    html.Div([  # Wrap the slider in a Div
        dcc.Slider(
        id='date-slider',
        min=0,
        max=len(date_indexes) - 1,
        value=0,
        marks={str(i): {'label': str(date_indexes[i].year) if hasattr(date_indexes[i], 'year') else str(date_indexes[i])[:4], 'style': {'writing-mode': 'vertical-rl'}} for i in range(0, len(date_indexes), 12)},
        step=1
        ),
    ], style={'margin-bottom': '20px'}),  # Add a bottom margin
    html.Div([  # Wrap the graphs in another Div
        dcc.Graph(id='graph-caaf-weights', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='graph-twostage-weights', style={'width': '50%', 'display': 'inline-block'}),
    ], style={'display': 'flex'}),  # Flex layout to arrange child Divs
    html.Div([
    dcc.Graph(id='line-chart-caaf-properties', style={'width': '50%', 'display': 'inline-block'}),
    dcc.Graph(id='line-chart-twostage-properties', style={'width': '50%', 'display': 'inline-block'}),
    ], style={'display': 'flex'}),
    html.Div([
    dcc.Graph(id='line-chart-caaf-diversification-measures', style={'width': '50%', 'display': 'inline-block'}),
    dcc.Graph(id='line-chart-twostage-diversification-measures', style={'width': '50%', 'display': 'inline-block'}),
], style={'display': 'flex'})
])

# Callback to update graph
@app.callback(
    Output('graph-efficient-frontier', 'figure'),
    [Input('date-slider', 'value')]
)
def update_graph(selected_date):
    selected_returns = block_expected_returns[selected_date]
    selected_stdevs = block_stdevs[selected_date]
    selected_weights = block_weights[selected_date]

    # Convert weights to string format
    weight_labels = [", ".join([f"{block.columns[i]}: {w[i]:.1%}" for i in range(len(w))]) for w in selected_weights]

    # Add data points from properties_caaf and properties_twostage for the selected date
    selected_date_str = str(date_indexes[selected_date])  # Assuming date_indexes are datetime objects
    caaf_point = properties_caaf.loc[selected_date_str, ['sigma', 'arithmetic_mu']]
    twostage_point = properties_twostage.loc[selected_date_str, ['sigma', 'arithmetic_mu']]

    caaf_frontier_for_date = efficient_frontier_current_caaf[efficient_frontier_current_caaf['Date'] == selected_date_str]
    twostage_frontier_for_date = efficient_frontier_two_stage[efficient_frontier_two_stage['Date'] == selected_date_str]

    caaf_point_weights = weight_caaf.loc[selected_date_str, :]
    twostage_point_weights = weight_twostage.loc[selected_date_str, :]

    caaf_weight_labels = ", ".join([f"{col}: {caaf_point_weights[col]:.1%}" for col in caaf_point_weights.index])
    twostage_weight_labels = ", ".join([f"{col}: {twostage_point_weights[col]:.1%}" for col in twostage_point_weights.index])

    # Create DataFrame for your main scatter plot
    df = pd.DataFrame({
        'Standard Deviation': selected_stdevs,
        'Expected Return': selected_returns,
        'Weights': weight_labels,
        'Color': ['rgba(64, 75, 151, 0.3)'] * len(selected_stdevs)  # 'blue' for normal points
    })

    # Create DataFrame for annotation points
    annotation_points = pd.DataFrame({
        'Standard Deviation': [caaf_point['sigma'], twostage_point['sigma']],
        'Expected Return': [caaf_point['arithmetic_mu'], twostage_point['arithmetic_mu']],
        'Weights': [caaf_weight_labels, twostage_weight_labels],  # Replace with actual weight or description
        'Color': ['rgba(216, 169, 23, 1)', 'rgba(160, 84, 66, 1)']  # 'red' and 'green' for annotation points
    })

    special_colors = annotation_points['Color'].tolist()
    special_points_trace = go.Scatter(
        x=annotation_points['Standard Deviation'],
        y=annotation_points['Expected Return'],
        mode='markers',
        name='Portfolio Methodology Points',
        marker=dict(
            color=['rgba(216, 169, 23, 1)', 'rgba(160, 84, 66, 1)'],  # Directly specify colors
            size=12
        ),
        hovertext=annotation_points['Weights']
    )

    # Combine the two DataFrames
    # combined_df = pd.concat([df, annotation_points], ignore_index=True)
    combined_df = df

    caaf_frontier_trace = go.Scatter(
        x=caaf_frontier_for_date['sigma'],
        y=caaf_frontier_for_date['arithmetic_mu'],
        mode='lines',
        name='CAAF Efficient Frontier',
        line=dict(color='blue', width=4)
    )
    twostage_frontier_trace = go.Scatter(
        x=twostage_frontier_for_date['sigma'],
        y=twostage_frontier_for_date['arithmetic_mu'],
        mode='lines',
        name='Two-Stage Efficient Frontier',
        line=dict(color='green', width=4)
    )

    # Create the scatter plot using combined DataFrame
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.6, 0.2, 0.2])
    # fig = px.scatter(combined_df, x='Standard Deviation', y='Expected Return', hover_data=['Weights'])
    main_points_trace = go.Scatter(
        x=combined_df['Standard Deviation'],
        y=combined_df['Expected Return'],
        mode='markers',
        name='Main Points',  # Unique name for legend
        marker=dict(color=hex_colors[0], size=8),
        hovertext=combined_df['Weights']
    )

    fig.add_trace(main_points_trace, row=1, col=1)
    fig.add_trace(caaf_frontier_trace, row=1, col=1)
    fig.add_trace(twostage_frontier_trace, row=1, col=1)
    fig.add_trace(go.Scatter(x=caaf_frontier_for_date["sigma"], y=caaf_frontier_for_date["enb"].values, mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=twostage_frontier_for_date["sigma"], y=twostage_frontier_for_date["enb"].values, mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=caaf_frontier_for_date["sigma"], y=caaf_frontier_for_date["adjusted_md"].values, mode='lines'), row=3, col=1)
    fig.add_trace(go.Scatter(x=twostage_frontier_for_date["sigma"], y=twostage_frontier_for_date["adjusted_md"].values, mode='lines'), row=3, col=1)
    fig.add_trace(special_points_trace, row=1, col=1)

    # Create a new marker color list
    # new_marker_colors = list(combined_df['Color'])

    # Update the trace
    # fig.update_traces(marker=dict(color=new_marker_colors, size=8))

    caaf_color = 'rgba(216, 169, 23, 1)'
    twostage_color = 'rgba(160, 84, 66, 1)'
    selected_date_only_str = selected_date_str.split(" ")[0]

    annotation_text = (
        f"<b>Special Points</b><br>"
        f"<span style='color:{caaf_color};'>■</span> CAAF Point<br>"
        f"<span style='color:{twostage_color};'>■</span> Two-Stage Point"
    )

    fig.add_annotation(
        text=annotation_text,
        x=1, y=1,  # Position in the upper right corner
        xref="paper", yref="paper",  # Positioning in paper coordinates
        showarrow=False,  # Don't show an arrow pointing from the text to a point
        align="right",  # Text alignment
        bgcolor="rgba(255, 255, 255, 0.8)",  # Background color with opacity
        bordercolor="black",  # Border color
        borderwidth=1,  # Border width
        borderpad=4  # Padding within the box
    )

    # Mimic Seaborn's whitegrid theme
    fig.update_layout(
        plot_bgcolor='white',  # White background
        xaxis=dict(
            gridcolor='lightgrey',  # X-axis grid lines
            showgrid=True,
            range=[min_stdev, max_stdev],
            title='Standard Deviation',
            tickformat='.1%'  # Formatting x-axis as percentage with one decimal place
        ),
        yaxis=dict(
            gridcolor='lightgrey',  # Y-axis grid lines
            showgrid=True,
            range=[min_expected_return, max_expected_return],
            title='Expected Return',
            tickformat='.1%'  # Formatting y-axis as percentage with one decimal place
        ),
    title=f'Efficient Frontier on {selected_date_only_str}'  # Adding title
    )
    return fig

# A function to update CAAF weight plot
def update_caaf_weights_plot():
    df = weight_caaf.reset_index()  # your dataframe
    fig_caaf = px.bar(df, x="Date", y=["Equities", "Gov Bonds", "Alternatives", "HY Credit", "Gold"], title="Current CAAF Weights", labels={'value': 'Asset Allocation'})
    fig_caaf.update_layout(bargap=0)
    for i, trace in enumerate(fig_caaf.data):
        trace_name = trace.name
        if trace_name in color_mapping:
            trace.marker.color = color_mapping[trace_name]
    return fig_caaf

# A function to update Two-Stage weight plot
def update_twostage_weights_plot():
    df = weight_twostage.reset_index()  # your dataframe
    fig_twostage = px.bar(df, x="Date", y=["Equities", "Gov Bonds", "Alternatives", "HY Credit", "Gold"], title="Two Stage Weights", labels={'value': 'Asset Allocation'})
    fig_twostage.update_layout(bargap=0)
    for i, trace in enumerate(fig_twostage.data):
        trace_name = trace.name
        if trace_name in color_mapping:
            trace.marker.color = color_mapping[trace_name]
    return fig_twostage

def create_caaf_properties_line_chart():
    df = properties_caaf.reset_index()  # Assuming 'Date' is the index
    fig = px.line(df, x='Date', y=['arithmetic_mu', 'sigma', 'naive_md', 'adjusted_md'])
    return fig

def create_twostage_properties_line_chart():
    df = properties_twostage.reset_index()  # Assuming 'Date' is the index
    fig = px.line(df, x='Date', y=['arithmetic_mu', 'sigma', 'naive_md', 'adjusted_md'])
    return fig

def create_caaf_diversification_measures_line_chart():
    df = properties_caaf.reset_index()  # Assuming 'Date' is the index
    fig = px.line(df, x='Date', y=['enb', 'div_ratio_sqrd'])
    fig.update_yaxes(range=[1, 4])
    return fig

def create_twostage_diversification_measures_line_chart():
    df = properties_twostage.reset_index()  # Assuming 'Date' is the index
    fig = px.line(df, x='Date', y=['enb', 'div_ratio_sqrd'])
    fig.update_yaxes(range=[1, 4])

    return fig

# On app initialization or as required
app.layout['graph-caaf-weights'].figure = update_caaf_weights_plot()
app.layout['graph-twostage-weights'].figure = update_twostage_weights_plot()
app.layout['line-chart-caaf-properties'].figure = create_caaf_properties_line_chart()
app.layout['line-chart-twostage-properties'].figure = create_twostage_properties_line_chart()
app.layout['line-chart-caaf-diversification-measures'].figure = create_caaf_diversification_measures_line_chart()
app.layout['line-chart-twostage-diversification-measures'].figure = create_twostage_diversification_measures_line_chart()

if __name__ == '__main__':
    app.run_server(debug=True)
