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
from tqdm import tqdm

import ffn
import bt
from reporting.tools.style import set_clbrm_style

from ffn.core import calc_two_stage_weights
from ffn.core import calc_current_caaf_weights
from ffn.core import calc_erc_weights

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

set_clbrm_style(caaf_colors=True)

import warnings
warnings.simplefilter(action='default', category=RuntimeWarning)

import pandas as pd

# The timestamp you used while saving
version_number = 2
target_version_number = 3
country = "US"
model = "Current CAAF"

# Read the DataFrames from the saved CSV files
weight_caaf = pd.read_csv(f'data/security_weights/Security_Weights_CurrentCAAF_{country}_{version_number}.csv', index_col=0)
weight_twostage = pd.read_csv(f'data/security_weights/Security_Weights_TwoStage_{country}_{version_number}.csv', index_col=0)
properties_caaf = pd.read_csv(f'data/portfolio_properties/Properties_CurrentCAAF_{country}_{version_number}.csv', index_col=0)
properties_twostage = pd.read_csv(f'data/portfolio_properties/Properties_TwoStage_{country}_{version_number}.csv', index_col=0)

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

er = pd.read_excel("./data/2023-10-26 master_file_US.xlsx", sheet_name="expected_gross_return")
er['Date'] = pd.to_datetime(er['Date'], format='%d/%m/%Y')
er.set_index('Date', inplace=True)

from concurrent.futures import ProcessPoolExecutor
import concurrent

def worker(date_chunk, rdf, er, const_covar):
    df_list_chunk = []
    for current_date in date_chunk:
        for target_volatility in np.arange(0.01, 0.20, 0.001):
            try:

                current_er_assets = er.loc[current_date, :].dropna().index

                erc_weights = calc_erc_weights(returns=rdf,
                                               initial_weights=None,
                                               risk_weights=None,
                                               covar_method="constant",
                                               risk_parity_method="slsqp",
                                               maximum_iterations=100,
                                               tolerance=1e-4,
                                               const_covar=const_covar.loc[current_er_assets, current_er_assets])
                if model == "Current CAAF":
                    weights, properties = calc_current_caaf_weights(returns=rdf,
                                                                    exp_rets=er.loc[current_date, :],
                                                                    target_volatility=target_volatility,
                                                                    erc_weights=erc_weights,
                                                                    covar_method="constant",
                                                                    const_covar=const_covar.loc[
                                                                    current_er_assets, current_er_assets],
                                                                    mode="frontier"
                                                                   )

                if model == "Two Stage":
                    weights, properties = calc_two_stage_weights(returns=rdf,
                                                                 exp_rets=er.loc[current_date, :],
                                                                 target_volatility=target_volatility,
                                                                 epsilon=0.1,
                                                                 erc_weights=erc_weights,
                                                                 covar_method="constant",
                                                                 const_covar=const_covar
                                                                )

                # Flatten twostage_weights to a dictionary
                weights_dict = weights.to_dict()

                # Flatten twostage_properties to a dictionary (also unpacking any nested lists)
                properties_dict = {}
                for key, value in properties.items():
                    if isinstance(value, np.ndarray) and value.shape[0] == 1:
                        properties_dict[key] = value[0]
                    elif isinstance(value, tuple):
                        properties_dict[key] = tuple(
                            v[0] if (isinstance(v, np.ndarray) and v.shape[0] == 1) else v for v in value)
                    else:
                        properties_dict[key] = value

                if 'md' in properties:
                    naive_md, adjusted_md = properties['md']

                    # Assuming each element of the tuple is an array of length 1
                    properties_dict['naive_md'] = naive_md[0]
                    properties_dict['adjusted_md'] = adjusted_md[0]

                    # Remove the original 'md' field
                    del properties_dict['md']

                # Combine all the dictionaries
                combined_dict = {
                    'Date': current_date,
                    'Target Volatility': target_volatility,
                    **weights_dict,
                    **properties_dict
                }

                df_current = pd.DataFrame([combined_dict])

                df_list_chunk.append(df_current)
            except Exception as e:
                if str(e) == 'Positive directional derivative for linesearch':
                    continue  # Skip the rest of the loop for this iteration
                else:
                    print(e)
    return df_list_chunk

if __name__ == '__main__':
    num_cores = 45
    # Splitting the dates into chunks. If there are 100 dates and you have 4 cores, each chunk might contain 25 dates.
    date_chunks = np.array_split(er.index, num_cores)

    # Use ProcessPoolExecutor to parallelize
    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_result = {executor.submit(worker, chunk, rdf, er, const_covar): chunk for chunk in date_chunks}
        for future in concurrent.futures.as_completed(future_to_result):
            df_list.extend(future.result())

    # Combine df_list into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_pickle(f"data/efficient_frontier_two_stage_{target_version_number}.pkl")

#    ##############################################################################
#    ##############################################################################

      # # Initialize empty list to collect DataFrames
      # df_list = []

      # # Loop through each date in er.index
      # for current_date in er.index:
      #     for target_volatility in np.arange(0.06, 0.12, 0.001):
      #         try:
      #             current_er_assets = er.loc[current_date, :].dropna().index

      #             erc_weights = calc_erc_weights(returns=rdf,
      #                                            initial_weights=None,
      #                                            risk_weights=None,
      #                                            covar_method="constant",
      #                                            risk_parity_method="slsqp",
      #                                            maximum_iterations=100,
      #                                            tolerance=1e-4,
      #                                            const_covar=const_covar.loc[current_er_assets, current_er_assets])

      #             # Similar code as in the worker function
      #             weights, properties = calc_current_caaf_weights(returns=rdf,
      #                                                             exp_rets=er.loc[current_date, :],
      #                                                             target_volatility=target_volatility,
      #                                                             erc_weights=erc_weights,
      #                                                             covar_method="constant",
      #                                                             const_covar=const_covar.loc[current_er_assets, current_er_assets]
      #                                                             )
      #             weights_dict = weights.to_dict()
      #             properties_dict = {}
      #             for key, value in properties.items():
      #                 if isinstance(value, np.ndarray) and value.shape[0] == 1:
      #                     properties_dict[key] = value[0]
      #                 elif isinstance(value, tuple):
      #                     properties_dict[key] = tuple(
      #                         v[0] if (isinstance(v, np.ndarray) and v.shape[0] == 1) else v for v in value)
      #                 else:
      #                     properties_dict[key] = value

      #             combined_dict = {
      #                 'Date': current_date,
      #                 'Target Volatility': target_volatility,
      #                 **weights_dict,
      #                 **properties_dict
      #             }

      #             if 'md' in properties:
      #                 naive_md, adjusted_md = properties['md']
      #                 properties_dict['naive_md'] = naive_md[0]
      #                 properties_dict['adjusted_md'] = adjusted_md[0]
      #                 del properties_dict['md']

      #             df_current = pd.DataFrame(combined_dict)
      #             df_list.append(df_current)

      #         except Exception as e:
      #             if str(e) == 'Positive directional derivative for linesearch':
      #                 continue
      #             else:
      #                 raise

      # # Concatenate all individual DataFrames to create the final DataFrame
      # final_df = pd.concat(df_list, ignore_index=True)

# # Save the DataFrame
# final_df.to_pickle("data/efficient_frontier_current_caaf.pkl")