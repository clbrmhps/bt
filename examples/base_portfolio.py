class BasePortfolio():
    def __init__(self, tickers=None, bounds=None, constraints=None):
        self.n_assets = len(tickers)
        self.tickers = tickers
        self.bounds = bounds
        self.constraints = constraints