import bt
import matplotlib.pyplot as plt

data = bt.get('spy,agg', start='2010-01-01')
print(data.head())

# create the strategy
s = bt.Strategy('s1', [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])

# create a backtest and run it
test = bt.Backtest(s, data)
res = bt.run(test)

# first let's see an equity curve
res.plot()
plt.show()

res.display()
res.plot_histogram()
plt.show()

res.plot_security_weights()
plt.show()

# create our new strategy
s2 = bt.Strategy('s2', [bt.algos.RunWeekly(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighInvVol(),
                        bt.algos.Rebalance()])

# now let's test it with the same data set. We will also compare it with our first backtest.
test2 = bt.Backtest(s2, data)
# we include test here to see the results side-by-side
res2 = bt.run(test, test2)
res2.plot()
plt.show()

res2.display()
