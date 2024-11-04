import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_properties(properties_folder, config_subset=None):
    property_data = {}

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

        for config_method, series in property_list:
            plt.plot(series.index, series, label=config_method)

        plt.xlabel('Date')
        plt.ylabel(property_name)
        plt.title(f'Comparison of {property_name} across Configurations and Methods')
        plt.legend()
        plt.grid(True)
        plt.show()

properties_folder = './properties'
config_subset = [19]
plot_properties(properties_folder, config_subset)
