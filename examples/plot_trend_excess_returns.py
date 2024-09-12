import pandas as pd
import matplotlib.pyplot as plt

from reporting.tools.style import set_clbrm_style

set_clbrm_style()

trend_data = pd.read_excel('./data/external_data/2023-10-04 AQR_TFR.xlsx', sheet_name='Returns')
trend_data.drop(columns=['Unnamed: 0'], inplace=True)
trend_data.rename(columns={'Unnamed: 1': 'Date'}, inplace=True)
trend_data = trend_data.iloc[:, :10]
trend_data.loc[:239, 'Date'] = pd.to_datetime(trend_data.loc[:239, 'Date'], format='%m/%d/%Y')
trend_data['Date'] = pd.to_datetime(trend_data['Date'], errors='coerce')

trend_data.set_index('Date', inplace=True)
trend_data.drop(trend_data.columns[[1, 3]], axis=1, inplace=True)
trend_data.columns = ['Base Excess Return', 'Cash', 'Total Return', 'Excess Return',
                      '20YMA Excess Return', 'US10Y Yield', 'Total Expected Return']

columns_to_plot = trend_data.columns[:3]
price_data = pd.DataFrame(index=trend_data.index, columns=columns_to_plot, data=1)
for column in columns_to_plot:
    price_data[column] = (1 + trend_data[column]).cumprod()
plt.figure(figsize=(12, 6))

custom_labels = {
    'Base Excess Return': 'AQR Trend-Following Excess Return',
    'Cash': 'AQR Cash (ICE BofAML 3-Month T-Bill)',
    'Total Return': 'AQR Trend-Following Total Return',
}

for column in columns_to_plot:
    plt.semilogy(price_data.index, price_data[column], label=custom_labels.get(column, column))

plt.title('Price Series from AQR')
plt.xlabel('Date')
plt.ylabel('Price (log scale)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

columns_to_plot = trend_data.columns[4:]
price_data = pd.DataFrame(index=trend_data.index, columns=columns_to_plot, data=1)
for column in columns_to_plot:
    price_data[column] = trend_data[column]
plt.figure(figsize=(12, 6))

custom_labels = {
    '20YMA Excess Return': '20-Year Moving Average Excess Return',
    'US10Y Yield': 'US 10-Year Government Bond Yield',
    'Total Expected Return': 'Total Expected Return',
}

for column in columns_to_plot:
    plt.plot(price_data.index, price_data[column], label=custom_labels.get(column, column))

plt.title('Constructed Expected Return Series')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
