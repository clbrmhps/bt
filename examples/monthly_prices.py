import pandas as pd
import numpy as np
import bt
import matplotlib.pyplot as plt

# 1. Generate synthetic monthly price data
np.random.seed(42)
date_rng = pd.date_range(start='1990-01-01', end='2020-12-31', freq='M')
num_assets = 3

prices = pd.DataFrame(index=date_rng)
for i in range(num_assets):
    price_changes = np.random.randn(len(date_rng)) * 0.02
    asset_prices = (1 + price_changes).cumprod() * 100  # start at 100
    prices[f'Asset_{i}'] = asset_prices

# 2. Implement ERC portfolio strategy
strategy = bt.Strategy('ERC_Portfolio',
                       [bt.algos.RunAfterDate('1996-01-01'),
                        bt.algos.RunQuarterly(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighERC(lookback=pd.DateOffset(months=3),
                                          covar_method='standard',
                                          maximum_iterations=10000,
                                          lag=pd.DateOffset(months=1)),
                        bt.algos.Rebalance()])

# 3. Backtest using bt
backtest = bt.Backtest(strategy, prices)
result = bt.run(backtest)

# Plot the results
result.plot()
plt.show()
