"""
A collection of Algos used to create Strategy logic.
"""
from __future__ import division

import abc
import random
import re

import numpy as np
import pandas as pd
import sklearn.covariance
from scipy.optimize import minimize

import datetime

import bt
from bt.core import Algo, AlgoStack, SecurityBase, is_zero

from analysis.drawdowns import endpoint_mdd_lookup

def pf_mu(weight, mu):
    # if weight.shape != mu.shape:
    #    raise Exception('Variables weight and mu must have same shape.')
    # weight = weight.reshape(-1, 1)
    # mu = mu.reshape(-1, 1)
    # return np.dot(weight.T, mu).ravel()
    return np.array([np.inner(weight, mu)])

def pf_sigma(weight, cov):
    # assert np.ndim(cov) == 2, 'Covariance matrix has to be 2-dimensional.'
    # assert np.max(weight.shape) == cov.shape[0] == cov.shape[1], 'Shapes of weight and cov are not aligned.'
    # sanity_checks.positive_semidefinite(matrix=cov)
    # weight = weight.reshape(-1, 1)
    # return np.sqrt(np.dot(weight.T, np.dot(cov, weight))).ravel()
    weight = weight.ravel()  # Flattening the weight array to 1D
    return np.sqrt(np.einsum('i,ij,j->', weight, cov, weight))

def pf_moments(weight, mu, is_geo, cov):
    assert isinstance(is_geo, bool), 'Variable is_geo has to be a boolean.'
    assert np.ndim(cov) == 2, 'Covariance matrix has to be 2-dimensional.'
    assert np.max(mu.shape) == cov.shape[0] == cov.shape[1],  'Shapes of mu and cov are not aligned.'
    if np.ndim(weight) == 1:
        assert weight.shape[0] == np.max(mu.shape), 'Shapes of weight and mu are not aligned.'
        weight = weight.reshape(1, -1)
    else:
        assert np.ndim(weight) == 2, 'Variables weight has to be 2-dimensional.'
        assert weight.shape[1] == np.max(mu.shape) != weight.shape[0], 'Variable weight cannot be a n x n matrix.'
    if is_geo:
        mu = mu + np.diag(cov) / 2
    pf_mu_ = np.full(weight.shape[0], np.nan)
    pf_sigma_ = np.full(weight.shape[0], np.nan)
    for i in range(weight.shape[0]):
        pf_mu_[i] = pf_mu(weight=weight[i, :], mu=mu.ravel())
        pf_sigma_[i] = pf_sigma(weight=weight[i, :], cov=cov)
    pf_geo_mu = pf_mu_ - np.square(pf_sigma_) / 2
    naive_md, adjusted_md = endpoint_mdd_lookup(geo_mu=pf_geo_mu, sigma=pf_sigma_, frequency='M',
                                                          percentile=5)
    return {'arithmetic_mu': pf_mu_,
            'geometric_mu': pf_geo_mu,
            'sigma': pf_sigma_,
            'naive_md': naive_md,
            'adjusted_md': adjusted_md}

def add_row_to_target_perm(tp, target_now, target_perm, properties=['arithmetic_mu', 'sigma', 'naive_md', 'adjusted_md', 'enb', 'div_ratio_sqrd']):
    new_row_data = {}

    for prop in properties:
        # Handle properties with nested structures differently
        if prop == 'md':
            naive_md, adjusted_md = tp.loc['md'][0][0], tp.loc['md'][1][0]
            new_row_data['naive_md'] = [naive_md]
            new_row_data['adjusted_md'] = [adjusted_md]
        else:
            value = tp.loc[prop] if prop in tp.index else None
            new_row_data[prop] = value

    new_row = pd.DataFrame(new_row_data, index=[target_now])

    if target_perm.empty:
        target_perm = new_row
    else:
        target_perm = pd.concat([target_perm, new_row])

    return target_perm

def filter_columns_based_return_availability(df, lookback=20):
    for col in df.columns:
        non_nan_count = df[col].count()

        if non_nan_count < lookback:
            df.drop(columns=[col], inplace=True)
    return df

def run_always(f):
    """
    Run always decorator to be used with Algo
    to ensure stack runs the decorated Algo
    on each pass, regardless of failures in the stack.
    """
    f.run_always = True
    return f

class PrintDate(Algo):

    """
    This Algo simply print's the current date.

    Can be useful for debugging purposes.
    """

    def __call__(self, target):
        print(target.now)
        return True


class PrintTempData(Algo):

    """
    This Algo prints the temp data.

    Useful for debugging.

    Args:
        * fmt_string (str): A string that will later be formatted with the
          target's temp dict. Therefore, you should provide
          what you want to examine within curly braces ( { } )
    """

    def __init__(self, fmt_string=None):
        super(PrintTempData, self).__init__()
        self.fmt_string = fmt_string

    def __call__(self, target):
        if self.fmt_string:
            print(self.fmt_string.format(**target.temp))
        else:
            print(target.temp)
        return True


class PrintInfo(Algo):

    """
    Prints out info associated with the target strategy. Useful for debugging
    purposes.

    Args:
        * fmt_string (str): A string that will later be formatted with the
          target object's __dict__ attribute. Therefore, you should provide
          what you want to examine within curly braces ( { } )

    Ex:
        PrintInfo('Strategy {name} : {now}')


    This will print out the name and the date (now) on each call.
    Basically, you provide a string that will be formatted with target.__dict__

    """

    def __init__(self, fmt_string="{name} {now}"):
        super(PrintInfo, self).__init__()
        self.fmt_string = fmt_string

    def __call__(self, target):
        print(self.fmt_string.format(**target.__dict__))
        return True


class Debug(Algo):

    """
    Utility Algo that calls pdb.set_trace when triggered.

    In the debug session, 'target' is available and can be examined through the
    StrategyBase interface.
    """

    def __call__(self, target):
        import pdb

        pdb.set_trace()
        return True


class RunOnce(Algo):

    """
    Returns True on first run then returns False.

    Args:
        * run_on_first_call: bool which determines if it runs the first time the algo is called

    As the name says, the algo only runs once. Useful in situations
    where we want to run the logic once (buy and hold for example).

    """

    def __init__(self):
        super(RunOnce, self).__init__()
        self.has_run = False

    def __call__(self, target):
        # if it hasn't run then we will
        # run it and set flag
        if not self.has_run:
            self.has_run = True
            return True

        # return false to stop future execution
        return False


class RunPeriod(Algo):
    def __init__(
        self, run_on_first_date=True, run_on_end_of_period=False, run_on_last_date=False
    ):
        super(RunPeriod, self).__init__()
        self._run_on_first_date = run_on_first_date
        self._run_on_end_of_period = run_on_end_of_period
        self._run_on_last_date = run_on_last_date

    def __call__(self, target):
        # get last date
        now = target.now

        # if none nothing to do - return false
        if now is None:
            return False

        # not a known date in our universe
        if now not in target.data.index:
            return False

        # get index of the current date
        index = target.data.index.get_loc(target.now)

        result = False

        # index 0 is a date added by the Backtest Constructor
        if index == 0:
            return False
        # first date
        if index == 1:
            if self._run_on_first_date:
                result = True
        # last date
        elif index == (len(target.data.index) - 1):
            if self._run_on_last_date:
                result = True
        else:
            # create pandas.Timestamp for useful .week,.quarter properties
            now = pd.Timestamp(now)

            index_offset = -1
            if self._run_on_end_of_period:
                index_offset = 1

            date_to_compare = target.data.index[index + index_offset]
            date_to_compare = pd.Timestamp(date_to_compare)

            result = self.compare_dates(now, date_to_compare)

        return result

    @abc.abstractmethod
    def compare_dates(self, now, date_to_compare):
        raise (NotImplementedError("RunPeriod Algo is an abstract class!"))


class RunDaily(RunPeriod):

    """
    Returns True on day change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's day has changed
    compared to the last(or next if run_on_end_of_period) date, if not returns False.
    Useful for daily rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.date() != date_to_compare.date():
            return True
        return False


class RunWeekly(RunPeriod):

    """
    Returns True on week change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's week has changed
    since relative to the last(or next) date, if not returns False. Useful for
    weekly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year or now.week != date_to_compare.week:
            return True
        return False


class RunMonthly(RunPeriod):

    """
    Returns True on month change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's month has changed
    since relative to the last(or next) date, if not returns False. Useful for
    monthly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year or now.month != date_to_compare.month:
            return True
        return False


class RunQuarterly(RunPeriod):

    """
    Returns True on quarter change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's quarter has changed
    since relative to the last(or next) date, if not returns False. Useful for
    quarterly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year or now.quarter != date_to_compare.quarter:
            return True
        return False


class RunYearly(RunPeriod):

    """
    Returns True on year change.

    Args:
        * run_on_first_date (bool): determines if it runs the first time the algo is called
        * run_on_end_of_period (bool): determines if it should run at the end of the period
          or the beginning
        * run_on_last_date (bool): determines if it runs on the last time the algo is called

    Returns True if the target.now's year has changed
    since relative to the last(or next) date, if not returns False. Useful for
    yearly rebalancing strategies.

    """

    def compare_dates(self, now, date_to_compare):
        if now.year != date_to_compare.year:
            return True
        return False


class RunOnDate(Algo):

    """
    Returns True on a specific set of dates.

    Args:
        * dates (list): List of dates to run Algo on.

    """

    def __init__(self, *dates):
        """
        Args:
            * dates (*args): A list of dates. Dates will be parsed
              by pandas.to_datetime so pass anything that it can
              parse. Typically, you will pass a string 'yyyy-mm-dd'.
        """
        super(RunOnDate, self).__init__()
        # parse dates and save
        self.dates = [pd.to_datetime(d) for d in dates]

    def __call__(self, target):
        return target.now in self.dates


class RunAfterDate(Algo):

    """
    Returns True after a date has passed

    Args:
        * date: Date after which to start trading

    Note:
        This is useful for algos that rely on trailing averages where you
        don't want to start trading until some amount of data has been built up

    """

    def __init__(self, date):
        """
        Args:
            * date: Date after which to start trading
        """
        super(RunAfterDate, self).__init__()
        # parse dates and save
        self.date = pd.to_datetime(date)

    def __call__(self, target):
        return target.now > self.date


class RunAfterDays(Algo):

    """
    Returns True after a specific number of 'warmup' trading days have passed

    Args:
        * days (int): Number of trading days to wait before starting

    Note:
        This is useful for algos that rely on trailing averages where you
        don't want to start trading until some amount of data has been built up

    """

    def __init__(self, days):
        """
        Args:
            * days (int): Number of trading days to wait before starting
        """
        super(RunAfterDays, self).__init__()
        self.days = days

    def __call__(self, target):
        if self.days > 0:
            self.days -= 1
            return False
        return True


class RunIfOutOfBounds(Algo):

    """
    This algo returns true if any of the target weights deviate by an amount greater
    than tolerance. For example, it will be run if the tolerance is set to 0.5 and
    a security grows from a target weight of 0.2 to greater than 0.3.

    A strategy where rebalancing is performed quarterly or whenever any
    security's weight deviates by more than 20% could be implemented by:

        Or([runQuarterlyAlgo,runIfOutOfBoundsAlgo(0.2)])

    Args:
        * tolerance (float): Allowed deviation of each security weight.

    Requires:
        * Weights

    """

    def __init__(self, tolerance):
        self.tolerance = float(tolerance)
        super(RunIfOutOfBounds, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        for cname in target.children:
            if cname in targets:
                c = target.children[cname]

                target_value = targets[cname]

                if np.isclose(target_value, 0, atol=1e-8) and np.isclose(c.weight, 0, atol=1e-8):
                    continue
                elif np.isclose(target_value, 0, atol=1e-8) and not np.isclose(c.weight, 0, atol=1e-8):
                    return True
                else:
                    deviation = abs((c.weight - target_value) / target_value)
                    if deviation > self.tolerance:
                        return True

        if "cash" in target.temp:
            cash_deviation = abs(
                (target.capital - targets.value) / targets.value - target.temp["cash"]
            )
            if cash_deviation > self.tolerance:
                return True

        return False

class RunIfOutOfTriggerThreshold(Algo):
    """
    This algo returns true if any of the target weights deviate by an amount greater
    than tolerance. For example, it will be run if the tolerance is set to 0.5 and
    a security grows from a target weight of 0.2 to greater than 0.3.

    A strategy where rebalancing is performed quarterly or whenever any
    security's weight deviates by more than 20% could be implemented by:

        Or([runQuarterlyAlgo,runIfOutOfBoundsAlgo(0.2)])

    Args:
        * tolerance (float): Allowed deviation of each security weight.

    Requires:
        * Weights

    """


    def __init__(self, tolerance):
        self.tolerance = float(tolerance)
        super(RunIfOutOfTriggerThreshold, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        mu_trigger = 0.001
        md_trigger = 0.01

        exp_rets = target.get_data('expected_returns').loc[target.now]
        exp_rets.dropna(inplace=True)

        const_covar = target.get_data('const_covar')

        target_weights = target.temp["weights"]
        current_weights = pd.Series({cname: target.children[cname].weight for cname in target.children})

        if len(current_weights) == 0:
            return True

        covar = const_covar.loc[list(exp_rets.index), list(exp_rets.index)]
        covar *= 12

        if current_weights.size != exp_rets.size:
            return True

        sorted_keys = sorted(exp_rets.index)
        # Sort exp_rets based on its own keys
        sorted_exp_rets = exp_rets.loc[sorted_keys]
        # Filter and sort only the keys present in exp_rets
        sorted_current_weights = current_weights.loc[current_weights.index.intersection(sorted_keys)].sort_index()
        sorted_target_weights = target_weights.loc[target_weights.index.intersection(sorted_keys)].sort_index()

        pf_moments_old = pf_moments(weight=sorted_current_weights.to_numpy(), mu=sorted_exp_rets.to_numpy(), is_geo=True, cov=covar.loc[sorted_keys, sorted_keys].to_numpy())
        pf_moments_new = pf_moments(weight=sorted_target_weights.to_numpy().astype(float), mu=sorted_exp_rets.to_numpy(), is_geo=True, cov=covar.loc[sorted_keys, sorted_keys].to_numpy())
        print(str(pf_moments_new['geometric_mu']) + ' vs ' + str(pf_moments_old['geometric_mu']))
        print(str(pf_moments_new['adjusted_md']) + ' vs ' + str(pf_moments_old['adjusted_md']))
        if (np.abs(pf_moments_new['adjusted_md']) < np.abs(pf_moments_old['adjusted_md']) - md_trigger) or \
                (pf_moments_new['geometric_mu'] > pf_moments_old['geometric_mu'] + mu_trigger):
            print('Update weights!')
            return True

        return False

class RunEveryNPeriods(Algo):

    """
    This algo runs every n periods.

    Args:
        * n (int): Run each n periods
        * offset (int): Applies to the first run. If 0, this algo will run the
          first time it is called.

    This Algo can be useful for the following type of strategy:
        Each month, select the top 5 performers. Hold them for 3 months.

    You could then create 3 strategies with different offsets and create a
    master strategy that would allocate equal amounts of capital to each.

    """

    def __init__(self, n, offset=0):
        super(RunEveryNPeriods, self).__init__()
        self.n = n
        self.offset = offset
        self.idx = n - offset - 1
        self.lcall = 0

    def __call__(self, target):
        # ignore multiple calls on same period
        if self.lcall == target.now:
            return False
        else:
            self.lcall = target.now
            # run when idx == (n-1)
            if self.idx == (self.n - 1):
                self.idx = 0
                return True
            else:
                self.idx += 1
                return False


class SelectAll(Algo):

    """
    Sets temp['selected'] with all securities (based on universe).

    Selects all the securities and saves them in temp['selected'].
    By default, SelectAll does not include securities that have no
    data (nan) on current date or those whose price is zero or negative.

    Args:
        * include_no_data (bool): Include securities that do not have data?
        * include_negative (bool): Include securities that have negative
          or zero prices?
    Sets:
        * selected

    """

    def __init__(self, include_no_data=False, include_negative=False):
        super(SelectAll, self).__init__()
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target):
        if self.include_no_data:
            target.temp["selected"] = target.universe.columns
        else:
            universe = target.universe.loc[target.now].dropna()
            if self.include_negative:
                target.temp["selected"] = list(universe.index)
            else:
                target.temp["selected"] = list(universe[universe > 0].index)
        return True


class SelectThese(Algo):

    """
    Sets temp['selected'] with a set list of tickers.

    Args:
        * ticker (list): List of tickers to select.
        * include_no_data (bool): Include securities that do not have data?
        * include_negative (bool): Include securities that have negative
          or zero prices?

    Sets:
        * selected

    """

    def __init__(self, tickers, include_no_data=False, include_negative=False):
        super(SelectThese, self).__init__()
        self.tickers = tickers
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target):
        if self.include_no_data:
            target.temp["selected"] = self.tickers
        else:
            universe = target.universe.loc[target.now, self.tickers].dropna()
            if self.include_negative:
                target.temp["selected"] = list(universe.index)
            else:
                target.temp["selected"] = list(universe[universe > 0].index)
        return True


class SelectHasData(Algo):

    """
    Sets temp['selected'] based on all items in universe that meet
    data requirements.

    This is a more advanced version of SelectAll. Useful for selecting
    tickers that need a certain amount of data for future algos to run
    properly.

    For example, if we need the items with 3 months of data or more,
    we could use this Algo with a lookback period of 3 months.

    When providing a lookback period, it is also wise to provide a min_count.
    This is basically the number of data points needed within the lookback
    period for a series to be considered valid. For example, in our 3 month
    lookback above, we might want to specify the min_count as being
    57 -> a typical trading month has give or take 20 trading days. If we
    factor in some holidays, we can use 57 or 58. It's really up to you.

    If you don't specify min_count, min_count will default to ffn's
    get_num_days_required.

    Args:
        * lookback (DateOffset): A DateOffset that determines the lookback
          period.
        * min_count (int): Minimum number of days required for a series to be
          considered valid. If not provided, ffn's get_num_days_required is
          used to estimate the number of points required.
        * include_no_data (bool): Include securities that do not have data?
        * include_negative (bool): Include securities that have negative
          or zero prices?
    Sets:
        * selected

    """

    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        min_count=None,
        include_no_data=False,
        include_negative=False,
    ):
        super(SelectHasData, self).__init__()
        self.lookback = lookback
        if min_count is None:
            min_count = bt.ffn.get_num_days_required(lookback)
        self.min_count = min_count
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target):
        if "selected" in target.temp:
            selected = target.temp["selected"]
        else:
            selected = target.universe.columns

        filt = target.universe.loc[target.now - self.lookback :, selected]
        cnt = filt.count()
        cnt = cnt[cnt >= self.min_count]
        if not self.include_no_data:
            cnt = cnt[~target.universe.loc[target.now, selected].isnull()]
            if not self.include_negative:
                cnt = cnt[target.universe.loc[target.now, selected] > 0]
        target.temp["selected"] = list(cnt.index)
        return True


class SelectN(Algo):

    """
    Sets temp['selected'] based on ranking temp['stat'].

    Selects the top or botton N items based on temp['stat'].
    This is usually some kind of metric that will be computed in a
    previous Algo and will be used for ranking purposes. Can select
    top or bottom N based on sort_descending parameter.

    Args:
        * n (int): select top n items.
        * sort_descending (bool): Should the stat be sorted in descending order
          before selecting the first n items?
        * all_or_none (bool): If true, only populates temp['selected'] if we
          have n items. If we have less than n, then temp['selected'] = [].
        * filter_selected (bool): If True, will only select from the existing
          'selected' list.

    Sets:
        * selected

    Requires:
        * stat

    """

    def __init__(
        self, n, sort_descending=True, all_or_none=False, filter_selected=False
    ):
        super(SelectN, self).__init__()
        if n < 0:
            raise ValueError("n cannot be negative")
        self.n = n
        self.ascending = not sort_descending
        self.all_or_none = all_or_none
        self.filter_selected = filter_selected

    def __call__(self, target):
        stat = target.temp["stat"].dropna()
        if self.filter_selected and "selected" in target.temp:
            stat = stat.loc[stat.index.intersection(target.temp["selected"])]
        stat.sort_values(ascending=self.ascending, inplace=True)

        # handle percent n
        keep_n = self.n
        if self.n < 1:
            keep_n = int(self.n * len(stat))

        sel = list(stat[:keep_n].index)

        if self.all_or_none and len(sel) < keep_n:
            sel = []

        target.temp["selected"] = sel

        return True


class SelectMomentum(AlgoStack):

    """
    Sets temp['selected'] based on a simple momentum filter.

    Selects the top n securities based on the total return over
    a given lookback period. This is just a wrapper around an
    AlgoStack with two algos: StatTotalReturn and SelectN.

    Note, that SelectAll() or similar should be called before
    SelectMomentum(), as StatTotalReturn uses values of temp['selected']

    Args:
        * n (int): select first N elements
        * lookback (DateOffset): lookback period for total return
          calculation
        * lag (DateOffset): Lag interval for total return calculation
        * sort_descending (bool): Sort descending (highest return is best)
        * all_or_none (bool): If true, only populates temp['selected'] if we
          have n items. If we have less than n, then temp['selected'] = [].

    Sets:
        * selected

    Requires:
        * selected

    """

    def __init__(
        self,
        n,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        sort_descending=True,
        all_or_none=False,
    ):
        super(SelectMomentum, self).__init__(
            StatTotalReturn(lookback=lookback, lag=lag),
            SelectN(n=n, sort_descending=sort_descending, all_or_none=all_or_none),
        )


class SelectWhere(Algo):

    """
    Selects securities based on an indicator DataFrame.

    Selects securities where the value is True on the current date
    (target.now) only if current date is present in signal DataFrame.

    For example, this could be the result of a pandas boolean comparison such
    as data > 100.

    Args:
        * signal (str|DataFrame): Boolean DataFrame containing selection logic.
          If a string is passed, frame is accessed using target.get_data
          This is the preferred way of using the algo.
        * include_no_data (bool): Include securities that do not have data?
        * include_negative (bool): Include securities that have negative
          or zero prices?

    Sets:
        * selected

    """

    def __init__(self, signal, include_no_data=False, include_negative=False):
        super(SelectWhere, self).__init__()
        if isinstance(signal, pd.DataFrame):
            self.signal_name = None
            self.signal = signal
        else:
            self.signal_name = signal
            self.signal = None

        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target):
        # get signal Series at target.now
        if self.signal_name is None:
            signal = self.signal
        else:
            signal = target.get_data(self.signal_name)

        if target.now in signal.index:
            sig = signal.loc[target.now]
            # get tickers where True
            # selected = sig.index[sig]
            selected = sig[sig == True].index  # noqa: E712
            # save as list
            if not self.include_no_data:
                universe = target.universe.loc[target.now, list(selected)].dropna()
                if self.include_negative:
                    selected = list(universe.index)
                else:
                    selected = list(universe[universe > 0].index)
            target.temp["selected"] = list(selected)

        return True


class SelectRandomly(AlgoStack):

    """
    Sets temp['selected'] based on a random subset of
    the items currently in temp['selected'].

    Selects n random elements from the list stored in temp['selected'].
    This is useful for benchmarking against a strategy where we believe
    the selection algorithm is adding value.

    For example, if we are testing a momentum strategy and we want to see if
    selecting securities based on momentum is better than just selecting
    securities randomly, we could use this Algo to create a random Strategy
    used for random benchmarking.

    Note:
        Another selection algorithm should be use prior to this Algo to
        populate temp['selected']. This will typically be SelectAll.

    Args:
        * n (int): Select N elements randomly.
        * include_no_data (bool): Include securities that do not have data?
        * include_negative (bool): Include securities that have negative
          or zero prices?

    Sets:
        * selected

    Requires:
        * selected

    """

    def __init__(self, n=None, include_no_data=False, include_negative=False):
        super(SelectRandomly, self).__init__()
        self.n = n
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target):
        if "selected" in target.temp:
            sel = target.temp["selected"]
        else:
            sel = list(target.universe.columns)

        if not self.include_no_data:
            universe = target.universe.loc[target.now, sel].dropna()
            if self.include_negative:
                sel = list(universe.index)
            else:
                sel = list(universe[universe > 0].index)

        if self.n is not None:
            n = self.n if self.n < len(sel) else len(sel)
            sel = random.sample(sel, int(n))

        target.temp["selected"] = sel
        return True


class SelectRegex(Algo):

    """
    Sets temp['selected'] based on a regex on their names.
    Useful when working with a large universe of different kinds of securities

    Args:
        * regex (str): regular expression on the name

    Sets:
        * selected

    Requires:
        * selected
    """

    def __init__(self, regex):
        super(SelectRegex, self).__init__()
        self.regex = re.compile(regex)

    def __call__(self, target):
        selected = target.temp["selected"]
        selected = [s for s in selected if self.regex.search(s)]
        target.temp["selected"] = selected
        return True


class ResolveOnTheRun(Algo):

    """
    Looks at securities set in temp['selected'] and searches for names that
    match the names of "aliases" for on-the-run securities in the provided
    data. Then replaces the alias with the name of the underlying security
    appropriate for the given date, and sets it back on temp['selected']

    Args:
        * on_the_run (str): Name of a Data frame with
            - columns set to "on the run" ticker names
            - index set to the timeline for the backtest
            - values are the actual security name to use for the given date
        * include_no_data (bool): Include securities that do not have data?
        * include_negative (bool): Include securities that have negative
          or zero prices?

    Requires:
        * selected

    Sets:
        * selected

    """

    def __init__(self, on_the_run, include_no_data=False, include_negative=False):
        super(ResolveOnTheRun, self).__init__()
        self.on_the_run = on_the_run
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target):
        # Resolve real tickers based on OTR
        on_the_run = target.get_data(self.on_the_run)
        selected = target.temp["selected"]
        aliases = [s for s in selected if s in on_the_run.columns]
        resolved = on_the_run.loc[target.now, aliases].tolist()
        if not self.include_no_data:
            universe = target.universe.loc[target.now, resolved].dropna()
            if self.include_negative:
                resolved = list(universe.index)
            else:
                resolved = list(universe[universe > 0].index)
        target.temp["selected"] = resolved + [
            s for s in selected if s not in on_the_run.columns
        ]
        return True


class SetStat(Algo):

    """
    Sets temp['stat'] for use by downstream algos (such as SelectN).

    Args:
        * stat (str|DataFrame): A dataframe of the same dimension as target.universe
          If a string is passed, frame is accessed using target.get_data
          This is the preferred way of using the algo.
    Sets:
        * stat
    """

    def __init__(self, stat):
        if isinstance(stat, pd.DataFrame):
            self.stat_name = None
            self.stat = stat
        else:
            self.stat_name = stat
            self.stat = None

    def __call__(self, target):
        if self.stat_name is None:
            stat = self.stat
        else:
            stat = target.get_data(self.stat_name)
        target.temp["stat"] = stat.loc[target.now]
        return True


class StatTotalReturn(Algo):

    """
    Sets temp['stat'] with total returns over a given period.

    Sets the 'stat' based on the total return of each element in
    temp['selected'] over a given lookback period. The total return
    is determined by ffn's calc_total_return.

    Args:
        * lookback (DateOffset): lookback period.
        * lag (DateOffset): Lag interval. Total return is calculated in
          the inteval [now - lookback - lag, now - lag]

    Sets:
        * stat

    Requires:
        * selected

    """

    def __init__(self, lookback=pd.DateOffset(months=3), lag=pd.DateOffset(days=0)):
        super(StatTotalReturn, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = target.temp["selected"]
        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        target.temp["stat"] = prc.calc_total_return()
        return True


class WeighEqually(Algo):

    """
    Sets temp['weights'] by calculating equal weights for all items in
    selected.

    Equal weight Algo. Sets the 'weights' to 1/n for each item in 'selected'.

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self):
        super(WeighEqually, self).__init__()

    def __call__(self, target):
        selected = target.temp["selected"]
        n = len(selected)

        if n == 0:
            target.temp["weights"] = {}
        else:
            w = 1.0 / n
            target.temp["weights"] = {x: w for x in selected}

        return True


class WeighSpecified(Algo):

    """
    Sets temp['weights'] based on a provided dict of ticker:weights.

    Sets the weights based on pre-specified targets.

    Args:
        * weights (dict): target weights -> ticker: weight

    Sets:
        * weights

    """

    def __init__(self, **weights):
        super(WeighSpecified, self).__init__()
        self.weights = weights

    def __call__(self, target):
        # added copy to make sure these are not overwritten
        target.temp["weights"] = self.weights.copy()
        return True


class ScaleWeights(Algo):

    """
    Sets temp['weights'] based on a scaled version of itself.
    Useful for going short, or scaling up/down when using
    :class:`FixedIncomeStrategy <bt.core.FixedIncomeStrategy>`.

    Args:
        * scale (float): the scaling factor

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, scale):
        super(ScaleWeights, self).__init__()
        self.scale = scale

    def __call__(self, target):
        target.temp["weights"] = {
            k: self.scale * w for k, w in target.temp["weights"].items()
        }
        return True


class WeighTarget(Algo):

    """
    Sets target weights based on a target weight DataFrame.

    If the target weight dataFrame is  of same dimension
    as the target.universe, the portfolio will effectively be rebalanced on
    each period. For example, if we have daily data and the target DataFrame
    is of the same shape, we will have daily rebalancing.

    However, if we provide a target weight dataframe that has only month end
    dates, then rebalancing only occurs monthly.

    Basically, if a weight is provided on a given date, the target weights are
    set and the algo moves on (presumably to a Rebalance algo). If not, not
    target weights are set.

    Args:
        * weights (str|DataFrame): DataFrame containing the target weights
          If a string is passed, frame is accessed using target.get_data
          This is the preferred way of using the algo.

    Sets:
        * weights

    """

    def __init__(self, weights):
        super(WeighTarget, self).__init__()
        if isinstance(weights, pd.DataFrame):
            self.weights_name = None
            self.weights = weights
        else:
            self.weights_name = weights
            self.weights = None

    def __call__(self, target):
        # get current target weights
        if self.weights_name is None:
            weights = self.weights
        else:
            weights = target.get_data(self.weights_name)

        if target.now in weights.index:
            w = weights.loc[target.now]

            # dropna and save
            target.temp["weights"] = w.dropna()

            return True
        else:
            return False


class WeighInvVol(Algo):

    """
    Sets temp['weights'] based on the inverse volatility Algo.

    Sets the target weights based on ffn's calc_inv_vol_weights. This
    is a commonly used technique for risk parity portfolios. The least
    volatile elements receive the highest weight under this scheme. Weights
    are proportional to the inverse of their volatility.

    Args:
        * lookback (DateOffset): lookback period for estimating volatility

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self, lookback=pd.DateOffset(months=3), lag=pd.DateOffset(days=0)):
        super(WeighInvVol, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        tw = bt.ffn.calc_inv_vol_weights(prc.to_returns().dropna())
        target.temp["weights"] = tw.dropna()
        return True


class WeighERC(Algo):

    """
    Sets temp['weights'] based on equal risk contribution algorithm.

    Sets the target weights based on ffn's calc_erc_weights. This
    is an extension of the inverse volatility risk parity portfolio in
    which the correlation of asset returns is incorporated into the
    calculation of risk contribution of each asset.

    The resulting portfolio is similar to a minimum variance portfolio
    subject to a diversification constraint on the weights of its components
    and its volatility is located between those of the minimum variance and
    equally-weighted portfolios (Maillard 2008).

    See:
        https://en.wikipedia.org/wiki/Risk_parity

    Args:
        * lookback (DateOffset): lookback period for estimating covariance
        * initial_weights (list): Starting asset weights [default inverse vol].
        * risk_weights (list): Risk target weights [default equal weight].
        * covar_method (str): method used to estimate the covariance. See ffn's
          calc_erc_weights for more details. (default ledoit-wolf).
        * risk_parity_method (str): Risk parity estimation method. see ffn's
          calc_erc_weights for more details. (default ccd).
        * maximum_iterations (int): Maximum iterations in iterative solutions
          (default 100).
        * tolerance (float): Tolerance level in iterative solutions (default 1E-8).


    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        initial_weights=None,
        risk_weights=None,
        covar_method="ledoit-wolf",
        risk_parity_method="ccd",
        maximum_iterations=100,
        tolerance=1e-8,
        lag=pd.DateOffset(days=0),
        additional_constraints=None,
        const_covar=None,
    ):
        super(WeighERC, self).__init__()
        self.lookback = lookback
        self.initial_weights = initial_weights
        self.risk_weights = risk_weights
        self.covar_method = covar_method
        self.risk_parity_method = risk_parity_method
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.lag = lag
        self.additional_constraints = additional_constraints
        self.const_covar = const_covar

    def __call__(self, target):
        selected = target.temp["selected"]

        if self.const_covar is None and self.covar_method == "constant":
            expected_returns = target.get_data('expected_returns')
            const_covar = target.get_data('const_covar')
            available_expected_returns = list(expected_returns.loc[target.now].dropna().index)

            const_covar = const_covar.loc[available_expected_returns, available_expected_returns]
        else:
            const_covar = self.const_covar

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        print(target.now)

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]

        prc = filter_columns_based_return_availability(prc)
        tw = bt.ffn.calc_erc_weights(
            prc.to_returns().dropna(),
            initial_weights=self.initial_weights,
            risk_weights=self.risk_weights,
            covar_method=self.covar_method,
            risk_parity_method=self.risk_parity_method,
            maximum_iterations=self.maximum_iterations,
            tolerance=self.tolerance,
            additional_constraints=self.additional_constraints,
            const_covar=const_covar,
        )

        target.temp["weights"] = tw.dropna()
        return True


class WeighMeanVar(Algo):

    """
    Sets temp['weights'] based on mean-variance optimization.

    Sets the target weights based on ffn's calc_mean_var_weights. This is a
    Python implementation of Markowitz's mean-variance optimization.

    See:
        http://en.wikipedia.org/wiki/Modern_portfolio_theory#The_efficient_frontier_with_no_risk-free_asset

    Args:
        * lookback (DateOffset): lookback period for estimating volatility
        * bounds ((min, max)): tuple specifying the min and max weights for
          each asset in the optimization.
        * covar_method (str): method used to estimate the covariance. See ffn's
          calc_mean_var_weights for more details.
        * rf (float): risk-free rate used in optimization.

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(
        self,
        expected_returns=None,
        lookback=pd.DateOffset(months=3),
        bounds=(0.0, 1.0),
        covar_method="ledoit-wolf",
        rf=0.0,
        lag=pd.DateOffset(days=0),
    ):
        super(WeighMeanVar, self).__init__()
        self.expected_returns = expected_returns
        self.lookback = lookback
        self.lag = lag
        self.bounds = bounds
        self.covar_method = covar_method
        self.rf = rf

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        tw = bt.ffn.calc_mean_var_weights_target_md(
            returns=prc.to_returns().dropna(),
            exp_rets=self.expected_returns.loc[target.now],
            target_md=0.30,
            weight_bounds=self.bounds,
            covar_method=self.covar_method,
        )

        target.temp["weights"] = tw.dropna()
        return True

class WeighTwoStage(Algo):
    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        initial_weights=None,
        risk_weights=None,
        covar_method="ledoit-wolf",
        risk_parity_method="ccd",
        maximum_iterations=100,
        tolerance=1e-8,
        lag=pd.DateOffset(days=0),
        bounds=(0.0, 1.0),
        additional_constraints=None,
        mode="short_term",
        return_factor=0.95,
        target_md=0.27
    ):
        super(WeighTwoStage, self).__init__()
        self.erc = WeighERC(
            lookback,
            initial_weights,
            risk_weights,
            covar_method,
            risk_parity_method,
            maximum_iterations,
            tolerance,
            lag,
        )
        self.lookback = lookback
        self.initial_weights = initial_weights
        self.risk_weights = risk_weights
        self.covar_method = covar_method
        self.risk_parity_method = risk_parity_method
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.lag = lag
        self.bounds = bounds
        self.additional_constraints = additional_constraints
        self.mode = mode
        self.return_factor = return_factor
        self.target_md = target_md

    def __call__(self, target):
        expected_returns = target.get_data('expected_returns')
        const_covar = target.get_data('const_covar')

        if self.target_md == "varying":
            target_md_var = target.get_data('target_md_var')
            target_md = target_md_var[target.now.date().strftime("%Y-%m-%d")]
        elif self.target_md == "frontier":
            efficient_frontier = target.get_data('efficient_frontier')
            target_md_var = target.get_data('target_md_var')
            target_md = target_md_var[target.now.date().strftime("%Y-%m-%d")]
            current_efficient_frontier = efficient_frontier.loc[target.now.date().strftime("%Y-%m-%d")].reset_index()
            current_efficient_frontier['abs_diff'] = abs(current_efficient_frontier['adjusted_md'] - target_md)
            min_diff_row = current_efficient_frontier.iloc[current_efficient_frontier['abs_diff'].idxmin()]
            closest_sigma = min_diff_row['sigma']
        elif self.target_md == "frontier_only":
                target_sigma = 0.07

                # Calculate the closest sigma on the Current CAAF Efficient Frontier
                efficient_frontier_current_caaf = target.get_data('efficient_frontier_current_caaf').reset_index()
                efficient_frontier_current_caaf_target_date = efficient_frontier_current_caaf.loc[efficient_frontier_current_caaf.loc[:, "Date"] == target.now]
                abs_diff_current_caaf = abs(efficient_frontier_current_caaf_target_date['sigma'] - target_sigma)
                min_diff_index_current_caaf = abs_diff_current_caaf.idxmin()
                closest_row_current_caaf = efficient_frontier_current_caaf.loc[min_diff_index_current_caaf]

                filtered_rows_current_caaf = efficient_frontier_current_caaf_target_date[efficient_frontier_current_caaf_target_date['sigma'] < target_sigma]
                higher_mu_rows_current_caaf = filtered_rows_current_caaf[filtered_rows_current_caaf['arithmetic_mu'] > closest_row_current_caaf['arithmetic_mu']]

                if not higher_mu_rows_current_caaf.empty:
                    max_mu_index_current_caaf = higher_mu_rows_current_caaf['arithmetic_mu'].idxmax()
                    closest_row_current_caaf = efficient_frontier_current_caaf.loc[max_mu_index_current_caaf]

                # Calculate the closest sigma on the Two Stage Efficient Frontier
                efficient_frontier = target.get_data('efficient_frontier').reset_index()
                efficient_frontier_target_date = efficient_frontier.loc[efficient_frontier.loc[:, "Date"] == target.now]
                abs_diff = abs(efficient_frontier_target_date['sigma'] - target_sigma)
                min_diff_index = abs_diff.idxmin()
                closest_row = efficient_frontier.loc[min_diff_index]

                filtered_rows = efficient_frontier_target_date[efficient_frontier_target_date['sigma'] < target_sigma]
                higher_mu_rows = filtered_rows[filtered_rows['arithmetic_mu'] > closest_row['arithmetic_mu']]

                if not higher_mu_rows.empty:
                    max_mu_index = higher_mu_rows['arithmetic_mu'].idxmax()
                    closest_row = efficient_frontier.loc[max_mu_index]

                # Adjust to Current CAAF Efficient Frontier if the difference is greater than 0.01
                if abs(closest_row_current_caaf['sigma'] - closest_row['sigma']) > 0.0025:
                    abs_diff = abs(efficient_frontier_target_date['sigma'] - closest_row_current_caaf['sigma'])
                    min_diff_index = abs_diff.idxmin()
                    closest_row = efficient_frontier.loc[min_diff_index]

                tp = pd.Series({
                    'arithmetic_mu': [closest_row['arithmetic_mu']],
                    'sigma': closest_row['sigma'],
                    'md': ([closest_row['naive_md']], [closest_row['adjusted_md']]),
                    'enb': closest_row['enb'],
                    'div_ratio_sqrd': closest_row['div_ratio_sqrd'],
                    'caaf_implied_epsilon': (closest_row['mv_arithmetic_mu'] - closest_row['arithmetic_mu'])/closest_row['mv_arithmetic_mu']
                })
                if "HY Credit" in closest_row.index:
                    tw = closest_row.loc[['Equities', 'Gov Bonds', 'HY Credit', 'Gold', 'Alternatives']]
                else:
                    tw = closest_row.loc[['Equities', 'Gov Bonds', 'Gold', 'Alternatives']]

                target.perm['properties'] = add_row_to_target_perm(tp, target.now, target.perm['properties'])
                target.temp["weights"] = tw.dropna()

                print(target.now)
                if target.now.date() == datetime.date(1885, 1, 31):
                    print("Break")

                return True
        else:
            target_md = self.target_md

        selected = target.temp["selected"]
        self.erc = WeighERC(
            self.lookback,
            self.initial_weights,
            self.risk_weights,
            self.covar_method,
            self.risk_parity_method,
            self.maximum_iterations,
            self.tolerance,
            self.lag,
            self.additional_constraints,
        )
        self.erc(target)
        erc_weights = target.temp["weights"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        print(target.now)
        if target.now.date() == datetime.date(1983, 2, 28):
            print("Break")

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        if self.mode == "long_term":
            prc = filter_columns_based_return_availability(prc)
        if self.target_md != "frontier":
            tw, tp = bt.ffn.calc_two_stage_weights(
                returns=prc.to_returns().dropna(),
                exp_rets=expected_returns.loc[target.now],
                target_md=target_md,
                epsilon=1-self.return_factor,
                erc_weights=erc_weights,
                weight_bounds=self.bounds,
                additional_constraints=self.additional_constraints,
                covar_method=self.covar_method,
                const_covar=const_covar
            )
        elif self.target_md == "frontier":
            tw, tp = bt.ffn.calc_two_stage_weights(
                returns=prc.to_returns().dropna(),
                exp_rets=expected_returns.loc[target.now],
                target_volatility=closest_sigma,
                epsilon=1-self.return_factor,
                erc_weights=erc_weights,
                weight_bounds=self.bounds,
                additional_constraints=self.additional_constraints,
                covar_method=self.covar_method,
                const_covar=const_covar
            )

        target.perm['properties'] = add_row_to_target_perm(tp, target.now, target.perm['properties'])
        target.temp["weights"] = tw.dropna()

        return True

class WeighMaxDiv(Algo):
    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        initial_weights=None,
        risk_weights=None,
        covar_method="ledoit-wolf",
        risk_parity_method="ccd",
        maximum_iterations=100,
        tolerance=1e-8,
        lag=pd.DateOffset(days=0),
        bounds=(0.0, 1.0),
        additional_constraints=None,
        mode="short_term",
        return_factor=0.95,
        target_md=0.4
    ):
        super(WeighMaxDiv, self).__init__()
        self.lookback = lookback
        self.initial_weights = initial_weights
        self.risk_weights = risk_weights
        self.covar_method = covar_method
        self.risk_parity_method = risk_parity_method
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.lag = lag
        self.bounds = bounds
        self.additional_constraints = additional_constraints
        self.mode = mode
        self.return_factor = return_factor
        self.target_md = target_md

    def __call__(self, target):
        expected_returns = target.get_data('expected_returns')
        const_covar = target.get_data('const_covar')

        selected = target.temp["selected"]
        available_expected_returns = list(expected_returns.loc[target.now].dropna().index)
        self.erc = WeighERC(
            self.lookback,
            self.initial_weights,
            self.risk_weights,
            self.covar_method,
            self.risk_parity_method,
            self.maximum_iterations,
            self.tolerance,
            self.lag,
            self.additional_constraints,
            const_covar.loc[available_expected_returns, available_expected_returns]
        )
        self.erc(target)
        erc_weights = target.temp["weights"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        print(target.now)
        if target.now.date() == datetime.date(1984, 4, 30):
            print("Break")

        t0 = target.now - self.lag

        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        if self.mode == "short_term":
            prc = filter_columns_based_return_availability(prc)

        ########################################################################
        # Test

        config = 2
        self.target_vol = 0.07

        year = target.now.year
        month = target.now.month
        day = target.now.day

        # Format into 'YYYYMMDD'
        formatted_date = f"{year}{month:02d}{day:02d}"
        model_data = pd.read_pickle(f"./data/frontiers/config_{config}/enb_{formatted_date}")

        closest_index = (model_data['Sigma'] - self.target_vol).abs().idxmin()
        closest_row = model_data.loc[closest_index]

        tp = pd.Series({
            'arithmetic_mu': [closest_row['Return']],
            'sigma': closest_row['Sigma'],
            'enb': closest_row['ENB'],
        })

        all_asset_classes = ['Equities', 'HY Credit', 'Gov Bonds', 'Gold', 'Alternatives']
        available_asset_classes = [column for column in all_asset_classes if column in closest_row.index]
        selected_asset_classes = list(closest_row[available_asset_classes].index)

        tw = closest_row[selected_asset_classes]
        tw = tw.apply(pd.to_numeric, errors='coerce')
        tw.name = 'MaxDiv'

        ########################################################################

        target.perm['properties'] = add_row_to_target_perm(tp, target.now, target.perm['properties'])
        target.temp["weights"] = tw.dropna()

        return True

class WeighCurrentCAAF(Algo):
    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        initial_weights=None,
        risk_weights=None,
        covar_method="ledoit-wolf",
        risk_parity_method="ccd",
        maximum_iterations=100,
        tolerance=1e-8,
        lag=pd.DateOffset(days=0),
        bounds=(0.0, 1.0),
        additional_constraints=None,
        mode="short_term",
        return_factor=0.95,
        target_md=0.4
    ):
        super(WeighCurrentCAAF, self).__init__()
        self.lookback = lookback
        self.initial_weights = initial_weights
        self.risk_weights = risk_weights
        self.covar_method = covar_method
        self.risk_parity_method = risk_parity_method
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.lag = lag
        self.bounds = bounds
        self.additional_constraints = additional_constraints
        self.mode = mode
        self.return_factor = return_factor
        self.target_md = target_md

    def __call__(self, target):
        expected_returns = target.get_data('expected_returns')
        const_covar = target.get_data('const_covar')

        selected = target.temp["selected"]
        available_expected_returns = list(expected_returns.loc[target.now].dropna().index)
        self.erc = WeighERC(
            self.lookback,
            self.initial_weights,
            self.risk_weights,
            self.covar_method,
            self.risk_parity_method,
            self.maximum_iterations,
            self.tolerance,
            self.lag,
            self.additional_constraints,
            const_covar.loc[available_expected_returns, available_expected_returns]
        )
        self.erc(target)
        erc_weights = target.temp["weights"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        print(target.now)
        if target.now.date() == datetime.date(1984, 4, 30):
            print("Break")

        t0 = target.now - self.lag

        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        if self.mode == "short_term":
            prc = filter_columns_based_return_availability(prc)

        tw, tp = bt.ffn.calc_current_caaf_weights(
            returns=prc.to_returns().dropna(),
            exp_rets=expected_returns.loc[target.now],
            target_md=self.target_md,
            erc_weights=erc_weights,
            weight_bounds=self.bounds,
            additional_constraints=self.additional_constraints,
            covar_method=self.covar_method,
            const_covar=const_covar,
            mode="optima"
        )

        target.perm['properties'] = add_row_to_target_perm(tp, target.now, target.perm['properties'])
        target.temp["weights"] = tw.dropna()

        return True

class WeighRandomly(Algo):

    """
    Sets temp['weights'] based on a random weight vector.

    Sets random target weights for each security in 'selected'.
    This is useful for benchmarking against a strategy where we believe
    the weighing algorithm is adding value.

    For example, if we are testing a low-vol strategy and we want to see if
    our weighing strategy is better than just weighing
    securities randomly, we could use this Algo to create a random Strategy
    used for random benchmarking.

    This is an Algo wrapper around ffn's random_weights function.

    Args:
        * bounds ((low, high)): Tuple including low and high bounds for each
          security
        * weight_sum (float): What should the weights sum up to?

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self, bounds=(0.0, 1.0), weight_sum=1):
        super(WeighRandomly, self).__init__()
        self.bounds = bounds
        self.weight_sum = weight_sum

    def __call__(self, target):
        sel = target.temp["selected"]
        n = len(sel)

        w = {}
        try:
            rw = bt.ffn.random_weights(n, self.bounds, self.weight_sum)
            w = dict(zip(sel, rw))
        except ValueError:
            pass

        target.temp["weights"] = w
        return True


class LimitDeltas(Algo):

    """
    Modifies temp['weights'] based on weight delta limits.

    Basically, this can be used if we want to restrict how much a security's
    target weight can change from day to day. Useful when we want to be more
    conservative about how much we could actually trade on a given day without
    affecting the market.

    For example, if we have a strategy that is currently long 100% one
    security, and the weighing Algo sets the new weight to 0%, but we
    use this Algo with a limit of 0.1, the new target weight will
    be 90% instead of 0%.

    Args:
        * limit (float, dict): Weight delta limit. If float, this will be a
          global limit for all securities. If dict, you may specify by-ticker
          limit.

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, limit=0.1):
        super(LimitDeltas, self).__init__()
        self.limit = limit
        # determine if global or specific
        self.global_limit = True
        if isinstance(limit, dict):
            self.global_limit = False

    def __call__(self, target):
        tw = target.temp["weights"]
        all_keys = set(list(target.children.keys()) + list(tw.keys()))

        for k in all_keys:
            tgt = tw[k] if k in tw else 0.0
            cur = target.children[k].weight if k in target.children else 0.0
            delta = tgt - cur

            # check if we need to limit
            if self.global_limit:
                if abs(delta) > self.limit:
                    tw[k] = cur + (self.limit * np.sign(delta))
            else:
                # make sure we have a limit defined in case of limit dict
                if k in self.limit:
                    lmt = self.limit[k]
                    if abs(delta) > lmt:
                        tw[k] = cur + (lmt * np.sign(delta))

        return True


class LimitWeights(Algo):

    """
    Modifies temp['weights'] based on weight limits.

    This is an Algo wrapper around ffn's limit_weights. The purpose of this
    Algo is to limit the weight of any one specifc asset. For example, some
    Algos will set some rather extreme weights that may not be acceptable.
    Therefore, we can use this Algo to limit the extreme weights. The excess
    weight is then redistributed to the other assets, proportionally to
    their current weights.

    See ffn's limit_weights for more information.

    Args:
        * limit (float): Weight limit.

    Sets:
        * weights

    Requires:
        * weights

    """

    def __init__(self, limit=0.1):
        super(LimitWeights, self).__init__()
        self.limit = limit

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        tw = target.temp["weights"]
        if len(tw) == 0:
            return True

        # if the limit < equal weight then set weights to 0
        if self.limit < 1.0 / len(tw):
            tw = {}
        else:
            tw = bt.ffn.limit_weights(tw, self.limit)
        target.temp["weights"] = tw

        return True


class TargetVol(Algo):
    """
    Updates temp['weights'] based on the target annualized volatility desired.

    Args:
        * target_volatility: annualized volatility to target
        * lookback (DateOffset): lookback period for estimating volatility
        * lag (DateOffset): amount of time to wait to calculate the covariance
        * covar_method: method of calculating volatility
        * annualization_factor: number of periods to annualize by.
          It is assumed that target volatility is already annualized by this factor.

    Updates:
        * weights

    Requires:
        * temp['weights']


    """

    def __init__(
        self,
        target_volatility,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        covar_method="standard",
        annualization_factor=252,
    ):
        super(TargetVol, self).__init__()
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.lag = lag
        self.covar_method = covar_method
        self.annualization_factor = annualization_factor

    def __call__(self, target):
        current_weights = target.temp["weights"]
        selected = current_weights.keys()

        # if there were no weights already set then skip
        if len(selected) == 0:
            return True

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        returns = bt.ffn.to_returns(prc)

        # calc covariance matrix
        if self.covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)
        elif self.covar_method == "standard":
            covar = returns.cov()
        else:
            raise NotImplementedError("covar_method not implemented")

        weights = pd.Series(
            [current_weights[x] for x in covar.columns], index=covar.columns
        )

        vol = np.sqrt(
            np.matmul(weights.values.T, np.matmul(covar.values, weights.values))
            * self.annualization_factor
        )

        if isinstance(self.target_volatility, (float, int)):
            self.target_volatility = {
                k: self.target_volatility for k in target.temp["weights"].keys()
            }

        for k in target.temp["weights"].keys():
            if k in self.target_volatility.keys():
                target.temp["weights"][k] = (
                    target.temp["weights"][k] * self.target_volatility[k] / vol
                )

        return True


class PTE_Rebalance(Algo):
    """
    Triggers a rebalance when PTE from static weights is past a level.

    Args:
        * PTE_volatility_cap: annualized volatility to target
        * target_weights: dataframe of weights that needs to have the same index as the price dataframe
        * lookback (DateOffset): lookback period for estimating volatility
        * lag (DateOffset): amount of time to wait to calculate the covariance
        * covar_method: method of calculating volatility
        * annualization_factor: number of periods to annualize by.
          It is assumed that target volatility is already annualized by this factor.

    """

    def __init__(
        self,
        PTE_volatility_cap,
        target_weights,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        covar_method="standard",
        annualization_factor=252,
    ):
        super(PTE_Rebalance, self).__init__()
        self.PTE_volatility_cap = PTE_volatility_cap
        self.target_weights = target_weights
        self.lookback = lookback
        self.lag = lag
        self.covar_method = covar_method
        self.annualization_factor = annualization_factor

    def __call__(self, target):
        if target.now is None:
            return False

        if target.positions.shape == (0, 0):
            return True

        positions = target.positions.loc[target.now]
        if positions is None:
            return True
        prices = target.universe.loc[target.now, positions.index]
        if prices is None:
            return True

        current_weights = positions * prices / target.value

        target_weights = self.target_weights.loc[target.now, :]

        cols = list(current_weights.index.copy())
        for c in target_weights.keys():
            if c not in cols:
                cols.append(c)

        weights = pd.Series(np.zeros(len(cols)), index=cols)
        for c in cols:
            if c in current_weights:
                weights[c] = current_weights[c]
            if c in target_weights:
                weights[c] -= target_weights[c]

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, cols]
        returns = bt.ffn.to_returns(prc)

        # calc covariance matrix
        if self.covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)
        elif self.covar_method == "standard":
            covar = returns.cov()
        else:
            raise NotImplementedError("covar_method not implemented")

        PTE_vol = np.sqrt(
            np.matmul(weights.values.T, np.matmul(covar.values, weights.values))
            * self.annualization_factor
        )

        if pd.isnull(PTE_vol):
            return False
        # vol is too high
        if PTE_vol > self.PTE_volatility_cap:
            return True
        else:
            return False

        return True


class CapitalFlow(Algo):

    """
    Used to model capital flows. Flows can either be inflows or outflows.

    This Algo can be used to model capital flows. For example, a pension
    fund might have inflows every month or year due to contributions. This
    Algo will affect the capital of the target node without affecting returns
    for the node.

    Since this is modeled as an adjustment, the capital will remain in the
    strategy until a re-allocation/rebalancement is made.

    Args:
        * amount (float): Amount of adjustment

    """

    def __init__(self, amount):
        """
        CapitalFlow constructor.

        Args:
            * amount (float): Amount to adjust by
        """
        super(CapitalFlow, self).__init__()
        self.amount = float(amount)

    def __call__(self, target):
        target.adjust(self.amount)
        return True


class CloseDead(Algo):

    """
    Closes all positions for which prices are equal to zero (we assume
    that these stocks are dead) and removes them from temp['weights'] if
    they enter it by any chance.
    To be called before Rebalance().

    In a normal workflow it is not needed, as those securities will not
    be selected by SelectAll(include_no_data=False) or similar method, and
    Rebalance() closes positions that are not in temp['weights'] anyway.
    However in case when for some reasons include_no_data=False could not
    be used or some modified weighting method is used, CloseDead() will
    allow to avoid errors.

    Requires:
        * weights

    """

    def __init__(self):
        super(CloseDead, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]
        for c in target.children:
            if target.universe[c].loc[target.now] <= 0:
                target.close(c)
                if c in targets:
                    del targets[c]

        return True


class SetNotional(Algo):

    """
    Sets the notional_value to use as the base for rebalancing for
    :class:`FixedIncomeStrategy <bt.core.FixedIncomeStrategy>` targets

    Args:
        * notional_value (str): Name of a pd.Series object containing the
          target notional values of the strategy over time.

    Sets:
        * notional_value
    """

    def __init__(self, notional_value):
        self.notional_value = notional_value
        super(SetNotional, self).__init__()

    def __call__(self, target):
        notional_value = target.get_data(self.notional_value)

        if target.now in notional_value.index:
            target.temp["notional_value"] = notional_value.loc[target.now]

            return True
        else:
            return False


class Rebalance(Algo):

    """
    Rebalances capital based on temp['weights']

    Rebalances capital based on temp['weights']. Also closes
    positions if open but not in target_weights. This is typically
    the last Algo called once the target weights have been set.

    Requires:
        * weights
        * cash (optional): You can set a 'cash' value on temp. This should be a
          number between 0-1 and determines the amount of cash to set aside.
          For example, if cash=0.3, the strategy will allocate 70% of its
          value to the provided weights, and the remaining 30% will be kept
          in cash. If this value is not provided (default), the full value
          of the strategy is allocated to securities.
        * notional_value (optional): Required only for fixed_income targets. This is the base
          balue of total notional that will apply to the weights.
    """

    def __init__(self):
        super(Rebalance, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        # save value because it will change after each call to allocate
        # use it as base in rebalance calls
        # call it before de-allocation so that notional_value is correct
        if target.fixed_income:
            if "notional_value" in target.temp:
                base = target.temp["notional_value"]
            else:
                base = target.notional_value
        else:
            base = target.value

        # de-allocate children that are not in targets and have non-zero value
        # (open positions)
        for cname in target.children:
            # if this child is in our targets, we don't want to close it out
            if cname in targets:
                continue

            # get child and value
            c = target.children[cname]
            if target.fixed_income:
                v = c.notional_value
            else:
                v = c.value

            # if non-zero and non-null, we need to close it out
            if v != 0.0 and not np.isnan(v):
                target.close(cname, update=False)

        # If cash is set (it should be a value between 0-1 representing the
        # proportion of cash to keep), calculate the new 'base'
        if "cash" in target.temp and not target.fixed_income:
            base = base * (1 - target.temp["cash"])

        # Turn off updating while we rebalance each child
        for item in targets.items():
            target.rebalance(item[1], child=item[0], base=base, update=False)

        # Now update
        target.root.update(target.now)

        return True


class RebalanceOverTime(Algo):

    """
    Similar to Rebalance but rebalances to target
    weight over n periods.

    Rebalances towards a target weight over a n periods. Splits up the weight
    delta over n periods.

    This can be useful if we want to make more conservative rebalacing
    assumptions. Some strategies can produce large swings in allocations. It
    might not be reasonable to assume that this rebalancing can occur at the
    end of one specific period. Therefore, this algo can be used to simulate
    rebalancing over n periods.

    This has typically been used in monthly strategies where we want to spread
    out the rebalancing over 5 or 10 days.

    Note:
        This Algo will require the run_always wrapper in the above case. For
        example, the RunMonthly will return True on the first day, and
        RebalanceOverTime will be 'armed'. However, RunMonthly will return
        False the rest days of the month. Therefore, we must specify that we
        want to always run this algo.

    Args:
        * n (int): number of periods over which rebalancing takes place.

    Requires:
        * weights

    """

    def __init__(self, n=10):
        super(RebalanceOverTime, self).__init__()
        self.n = float(n)
        self._rb = Rebalance()
        self._weights = None
        self._days_left = None

    def __call__(self, target):
        # new weights specified - update rebalance data
        if "weights" in target.temp:
            self._weights = target.temp["weights"]
            self._days_left = self.n

        # if _weights are not None, we have some work to do
        if self._weights is not None:
            tgt = {}
            # scale delta relative to # of periods left and set that as the new
            # target
            for cname in self._weights.keys():
                curr = (
                    target.children[cname].weight if cname in target.children else 0.0
                )
                dlt = (self._weights[cname] - curr) / self._days_left
                tgt[cname] = curr + dlt

            # mock weights and call real Rebalance
            target.temp["weights"] = tgt
            self._rb(target)

            # dec _days_left. If 0, set to None & set _weights to None
            self._days_left -= 1

            if self._days_left == 0:
                self._days_left = None
                self._weights = None

        return True

class Require(Algo):

    """
    Flow control Algo.

    This algo returns the value of a predicate
    on an temp entry. Useful for controlling
    flow.

    For example, we might want to make sure we have some items selected.
    We could pass a lambda function that checks the len of 'selected':

        pred=lambda x: len(x) == 0
        item='selected'

    Args:
        * pred (Algo): Function that returns a Bool given the strategy. This
          is the definition of an Algo. However, this is typically used
          with a simple lambda function.
        * item (str): An item within temp.
        * if_none (bool): Result if the item required is not in temp or if it's
          value if None

    """

    def __init__(self, pred, item, if_none=False):
        super(Require, self).__init__()
        self.item = item
        self.pred = pred
        self.if_none = if_none

    def __call__(self, target):
        if self.item not in target.temp:
            return self.if_none

        item = target.temp[self.item]

        if item is None:
            return self.if_none

        return self.pred(item)


class Not(Algo):
    """
    Flow control Algo

    It is usful for "inverting" other flow control algos,
    For example Not( RunAfterDate(...) ), Not( RunAfterDays(...) ), etc

    Args:
        * list_of_algos (Algo): The algo to run and invert the return value of
    """

    def __init__(self, algo):
        super(Not, self).__init__()
        self._algo = algo

    def __call__(self, target):
        return not self._algo(target)


class Or(Algo):
    """
    Flow control Algo

    It useful for combining multiple signals into one signal.
    For example, we might want two different rebalance signals to work together:

        runOnDateAlgo = bt.algos.RunOnDate(pdf.index[0]) # where pdf.index[0] is the first date in our time series
        runMonthlyAlgo = bt.algos.RunMonthly()
        orAlgo = Or([runMonthlyAlgo,runOnDateAlgo])

    orAlgo will return True if it is the first date or if it is 1st of the month

    Args:
        * list_of_algos: Iterable list of algos.
          Runs each algo and
          returns true if any algo returns true.
    """

    def __init__(self, list_of_algos):
        super(Or, self).__init__()
        self._list_of_algos = list_of_algos
        return

    def __call__(self, target):
        res = False
        for algo in self._list_of_algos:
            tempRes = algo(target)
            res = res | tempRes

        return res

class And(Algo):
    """
    Flow control Algo

    It useful for combining multiple signals into one signal.
    For example, we might want two different rebalance signals to work together:

        runOnDateAlgo = bt.algos.RunOnDate(pdf.index[0]) # where pdf.index[0] is the first date in our time series
        runMonthlyAlgo = bt.algos.RunMonthly()
        andAlgo = And([runMonthlyAlgo,runOnDateAlgo])

    andAlgo will return True if it is the first date and if it is 1st of the month

    Args:
        * list_of_algos: Iterable list of algos.
          Runs each algo and
          returns true if all algos return true.
    """

    def __init__(self, list_of_algos):
        super(And, self).__init__()
        self._list_of_algos = list_of_algos
        return

    def __call__(self, target):
        res = True
        for algo in self._list_of_algos:
            tempRes = algo(target)
            res = res & tempRes

        return res

class SelectTypes(Algo):
    """
    Sets temp['selected'] based on node type.
    If temp['selected'] is already set, it will filter the existing
    selection.

    Args:
        * include_types (list): Types of nodes to include
        * exclude_types (list): Types of nodes to exclude

    Sets:
        * selected
    """

    def __init__(self, include_types=(bt.core.Node,), exclude_types=()):
        super(SelectTypes, self).__init__()
        self.include_types = include_types
        self.exclude_types = exclude_types or (type(None),)

    def __call__(self, target):
        selected = [
            sec_name
            for sec_name, sec in target.children.items()
            if isinstance(sec, self.include_types)
            and not isinstance(sec, self.exclude_types)
        ]
        if "selected" in target.temp:
            selected = [s for s in selected if s in target.temp["selected"]]
        target.temp["selected"] = selected
        return True


class ClosePositionsAfterDates(Algo):

    """
    Close positions on securities after a given date.
    This can be used to make sure positions on matured/redeemed securities are
    closed. It can also be used as part of a strategy to, i.e. make sure
    the strategy doesn't hold any securities with time to maturity less than a year

    Note that if placed after a RunPeriod algo in the stack, that the actual
    closing of positions will occur after the provided date. For this to work,
    the "price" of the security (even if matured) must exist up until that date.
    Alternatively, run this with the @run_always decorator to close the positions
    immediately.

    Also note that this algo does not operate using temp['weights'] and Rebalance.
    This is so that hedges (which are excluded from that workflow) will also be
    closed as necessary.

    Args:
        * close_dates (str): the name of a dataframe indexed by security name, with columns
          "date": the date after which we want to close the position ASAP

    Sets:
        * target.perm['closed'] : to keep track of which securities have already closed
    """

    def __init__(self, close_dates):
        super(ClosePositionsAfterDates, self).__init__()
        self.close_dates = close_dates

    def __call__(self, target):
        if "closed" not in target.perm:
            target.perm["closed"] = set()
        close_dates = target.get_data(self.close_dates)["date"]
        # Find securities that are candidate for closing
        sec_names = [
            sec_name
            for sec_name, sec in target.children.items()
            if isinstance(sec, SecurityBase)
            and sec_name in close_dates.index
            and sec_name not in target.perm["closed"]
        ]

        # Check whether closed
        is_closed = close_dates.loc[sec_names] <= target.now

        # Close position
        for sec_name in is_closed[is_closed].index:
            target.close(sec_name, update=False)
            target.perm["closed"].add(sec_name)

        # Now update
        target.root.update(target.now)

        return True


class RollPositionsAfterDates(Algo):

    """
    Roll securities based on the provided map.
    This can be used for any securities which have "On-The-Run" and "Off-The-Run"
    versions (treasury bonds, index swaps, etc).

    Also note that this algo does not operate using temp['weights'] and Rebalance.
    This is so that hedges (which are excluded from that workflow) will also be
    rolled as necessary.

    Args:
        * roll_data (str): the name of a dataframe indexed by security name, with columns
            - "date": the first date at which the roll can occur
            - "target": the security name we are rolling into
            - "factor": the conversion factor. One unit of the original security
              rolls into "factor" units of the new one.

    Sets:
        * target.perm['rolled'] : to keep track of which securities have already rolled
    """

    def __init__(self, roll_data):
        super(RollPositionsAfterDates, self).__init__()
        self.roll_data = roll_data

    def __call__(self, target):
        if "rolled" not in target.perm:
            target.perm["rolled"] = set()
        roll_data = target.get_data(self.roll_data)
        transactions = {}
        # Find securities that are candidate for roll
        sec_names = [
            sec_name
            for sec_name, sec in target.children.items()
            if isinstance(sec, SecurityBase)
            and sec_name in roll_data.index
            and sec_name not in target.perm["rolled"]
        ]

        # Calculate new transaction and close old position
        for sec_name, sec_fields in roll_data.loc[sec_names].iterrows():
            if sec_fields["date"] <= target.now:
                target.perm["rolled"].add(sec_name)
                new_quantity = sec_fields["factor"] * target[sec_name].position
                new_sec = sec_fields["target"]
                if new_sec in transactions:
                    transactions[new_sec] += new_quantity
                else:
                    transactions[new_sec] = new_quantity
                target.close(sec_name, update=False)

        # Do all the new transactions at the end, to do any necessary aggregations first
        for new_sec, quantity in transactions.items():
            target.transact(quantity, new_sec, update=False)

        # Now update
        target.root.update(target.now)

        return True


class SelectActive(Algo):

    """
    Sets temp['selected'] based on filtering temp['selected'] to exclude
    those securities that have been closed or rolled after a certain date
    using ClosePositionsAfterDates or RollPositionsAfterDates. This makes sure
    not to select them again for weighting (even if they have prices).

    Requires:
        * selected
        * perm['closed'] or perm['rolled']

    Sets:
        * selected

    """

    def __call__(self, target):
        selected = target.temp["selected"]
        rolled = target.perm.get("rolled", set())
        closed = target.perm.get("closed", set())
        selected = [s for s in selected if s not in set.union(rolled, closed)]
        target.temp["selected"] = selected
        return True


class ReplayTransactions(Algo):

    """
    Replay a list of transactions that were executed.
    This is useful for taking a blotter of actual trades that occurred,
    and measuring performance against hypothetical strategies.
    In particular, one can replay the outputs of backtest.Result.get_transactions

    Note that this allows the timestamps and prices of the reported transactions
    to be completely arbitrary, so while the strategy may track performance
    on a daily basis, it will accurately account for the actual PNL of
    the trades based on where they actually traded, and the bidofferpaid
    attribute on the strategy will capture the "slippage" as measured
    against the daily prices.

    Args:
        * transactions (str): name of a MultiIndex dataframe with format
          Date, Security | quantity, price.
          Note this schema follows the output of backtest.Result.get_transactions

    """

    def __init__(self, transactions):
        super(ReplayTransactions, self).__init__()
        self.transactions = transactions

    def __call__(self, target):
        timeline = target.data.index
        index = timeline.get_loc(target.now)
        end = target.now
        if index == 0:
            start = pd.Timestamp.min
        else:
            start = timeline[index - 1]
        # Get the transactions since the last update
        all_transactions = target.get_data(self.transactions)
        timestamps = all_transactions.index.get_level_values("Date")
        transactions = all_transactions[(timestamps > start) & (timestamps <= end)]
        for (_, security), transaction in transactions.iterrows():
            c = target[security]
            c.transact(
                transaction["quantity"], price=transaction["price"], update=False
            )

        # Now update
        target.root.update(target.now)

        return True


class SimulateRFQTransactions(Algo):
    """
    An algo that simulates the outcomes from RFQs (Request for Quote)
    using a "model" that determines which ones becomes transactions and at what price
    those transactions happen. This can be used from the perspective of the sender of the
    RFQ or the receiver.

    Args:
        * rfqs (str): name of a dataframe with columns
          Date, Security | quantity, *additional columns as required by model
        * model (object): a function/callable object with arguments
                - rfqs : data frame of rfqs to respond to
                - target : the strategy object, for access to position and value data
          and which returns a set of transactions, a MultiIndex DataFrame with:
                Date, Security | quantity, price
    """

    def __init__(self, rfqs, model):
        super(SimulateRFQTransactions, self).__init__()
        self.rfqs = rfqs
        self.model = model

    def __call__(self, target):
        timeline = target.data.index
        index = timeline.get_loc(target.now)
        end = target.now
        if index == 0:
            start = pd.Timestamp.min
        else:
            start = timeline[index - 1]
        # Get the RFQs since the last update
        all_rfqs = target.get_data(self.rfqs)
        timestamps = all_rfqs.index.get_level_values("Date")
        rfqs = all_rfqs[(timestamps > start) & (timestamps <= end)]

        # Turn the RFQs into transactions
        transactions = self.model(rfqs, target)

        for (_, security), transaction in transactions.iterrows():
            c = target[security]
            c.transact(
                transaction["quantity"], price=transaction["price"], update=False
            )

        # Now update
        target.root.update(target.now)

        return True


def _get_unit_risk(security, data, index=None):
    try:
        unit_risks = data[security]
        unit_risk = unit_risks.values[index]
    except Exception:
        # No risk data, assume zero
        unit_risk = 0.0
    return unit_risk


class UpdateRisk(Algo):

    """
    Tracks a risk measure on all nodes of the strategy. To use this node, the
    ``additional_data`` argument on :class:`Backtest <bt.backtest.Backtest>` must
    have a "unit_risk" key. The value should be a dictionary, keyed
    by risk measure, of DataFrames with a column per security that is sensitive to that measure.

    Args:
        * name (str): the name of the risk measure (IR01, PVBP, IsIndustials, etc).
          The name must coincide with the keys of the dictionary passed to additional_data as the
          "unit_risk" argument.
        * history (int): The level of depth in the tree at which to track the time series of risk numbers.
          i.e. 0=no tracking, 1=first level only, etc. More levels is more expensive.

    Modifies:
        * The "risk" attribute on the target and all its children
        * If history==True, the "risks" attribute on the target and all its children

    """

    def __init__(self, measure, history=0):
        super(UpdateRisk, self).__init__(name="UpdateRisk>%s" % measure)
        self.measure = measure
        self.history = history

    def _setup_risk(self, target, set_history):
        """Setup risk attributes on the node in question"""
        target.risk = {}
        if set_history:
            target.risks = pd.DataFrame(index=target.data.index)

    def _setup_measure(self, target, set_history):
        """Setup a risk measure within the risk attributes on the node in question"""
        target.risk[self.measure] = np.NaN
        if set_history:
            target.risks[self.measure] = np.NaN

    def _set_risk_recursive(self, target, depth, unit_risk_frame):
        set_history = depth < self.history
        # General setup of risk on nodes
        if not hasattr(target, "risk"):
            self._setup_risk(target, set_history)
        if self.measure not in target.risk:
            self._setup_measure(target, set_history)

        if isinstance(target, bt.core.SecurityBase):
            # Use target.root.now as non-traded securities may not have been updated yet
            # and there is no need to update them here as we only use position
            index = unit_risk_frame.index.get_loc(target.root.now)
            unit_risk = _get_unit_risk(target.name, unit_risk_frame, index)
            if is_zero(target.position):
                risk = 0.0
            else:
                risk = unit_risk * target.position * target.multiplier
        else:
            risk = 0.0
            for child in target.children.values():
                self._set_risk_recursive(child, depth + 1, unit_risk_frame)
                risk += child.risk[self.measure]

        target.risk[self.measure] = risk
        if depth < self.history:
            target.risks.loc[target.now, self.measure] = risk

    def __call__(self, target):
        unit_risk_frame = target.get_data("unit_risk")[self.measure]
        self._set_risk_recursive(target, 0, unit_risk_frame)
        return True


class PrintRisk(Algo):

    """
    This Algo prints the risk data.

    Args:
        * fmt_string (str): A string that will later be formatted with the
          target object's risk attributes. Therefore, you should provide
          what you want to examine within curly braces ( { } )
          If not provided, will print the entire dictionary with no formatting.
    """

    def __init__(self, fmt_string=""):
        super(PrintRisk, self).__init__()
        self.fmt_string = fmt_string

    def __call__(self, target):
        if hasattr(target, "risk"):
            if self.fmt_string:
                print(self.fmt_string.format(**target.risk))
            else:
                print(target.risk)
        return True


class HedgeRisks(Algo):
    """
    Hedges risk measures with selected instruments.

    Make sure that the UpdateRisk algo has been called beforehand.

    Args:
        * measures (list): the names of the risk measures to hedge
        * pseudo (bool): whether to use the pseudo-inverse to compute
          the inverse Jacobian. If False, will fail if the number
          of selected instruments is not equal to the number of
          measures, or if the Jacobian is singular
        * strategy (StrategyBase): If provided, will hedge the risk
          from this strategy in addition to the risk from target.
          This is to allow separate tracking of hedged and unhedged
          performance. Note that risk_strategy must occur earlier than
          'target' in a depth-first traversal of the children of the root,
          otherwise hedging will occur before positions of risk_strategy are
          updated.
        * throw_nan (bool): Whether to throw on nan hedge notionals, rather
          than simply not hedging.

    Requires:
        * selected
    """

    def __init__(self, measures, pseudo=False, strategy=None, throw_nan=True):
        super(HedgeRisks, self).__init__()
        if len(measures) == 0:
            raise ValueError("Must pass in at least one measure to hedge")
        self.measures = measures
        self.pseudo = pseudo
        self.strategy = strategy
        self.throw_nan = throw_nan

    def _get_target_risk(self, target, measure):
        if not hasattr(target, "risk"):
            raise ValueError("risk not set up on target %s" % target.name)
        if measure not in target.risk:
            raise ValueError("measure %s not set on target %s" % (measure, target.name))
        return target.risk[measure]

    def __call__(self, target):
        securities = target.temp["selected"]

        # Get target risk
        target_risk = np.array(
            [self._get_target_risk(target, m) for m in self.measures]
        )
        if self.strategy is not None:
            # Add the target risk of the strategy to the risk of the target
            # (which contains existing hedges)
            target_risk += np.array(
                [self._get_target_risk(self.strategy, m) for m in self.measures]
            )
        # Turn target_risk into a column array
        target_risk = target_risk.reshape(len(self.measures), 1)

        # Get hedge risk as a Jacobian matrix
        data = []
        for m in self.measures:
            d = target.get_data("unit_risk").get(m)
            if d is None:
                raise ValueError(
                    "unit_risk for %s not present in temp on %s"
                    % (self.measure, target.name)
                )
            i = d.index.get_loc(target.now)
            data.append((i, d))

        hedge_risk = np.array(
            [[_get_unit_risk(s, d, i) for (i, d) in data] for s in securities]
        )

        # Get hedge ratios
        if self.pseudo:
            inv = np.linalg.pinv(hedge_risk).T
        else:
            inv = np.linalg.inv(hedge_risk).T
        notionals = np.matmul(inv, -target_risk).flatten()

        # Hedge
        for notional, security in zip(notionals, securities):
            if np.isnan(notional) and self.throw_nan:
                raise ValueError("%s has nan hedge notional" % security)
            target.transact(notional, security)
        return True

class ExpectedReturns(Algo):
    def __init__(self, expected_returns):
        self.expected_returns = expected_returns

    def __call__(self, target):
        return True

class ConstantCovar(Algo):
    def __init__(self, const_covar):
        self.const_covar = const_covar

    def __call__(self, target):
        return True