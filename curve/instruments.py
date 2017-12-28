# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np

class Instrument(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def par_rate(self, curve):
        pass
    
class SimpleRate(Instrument):
    def __init__(self, start, end):
        self._start = start
        self._end = end
    
    def par_rate(self, curve):
        df_start = curve.get_df(self._start)
        df_end = curve.get_df(self._end)
        return (df_start / df_end - 1) / (self._end - self._start)

class SingleCurrencySwap(Instrument):
    def __init__(self, start, end, roll):
        self._start = start
        self._end = end
        self._roll = roll
        
    def par_rate(self, curve):
        df_start = curve.get_df(self._start)
        df_end = curve.get_df(self._end)
        # simple implementation for ease
        def generate_payment_dates(start, end, roll):
            return np.arange(start, end, roll) + roll
        payment_dates = generate_payment_dates(self._start,
                                               self._end,
                                               self._roll)
        annuity = np.sum(curve.get_df(payment_dates)) * self._roll
        return (df_start - df_end) / annuity

class ZeroRate(Instrument):
    def __init__(self, start, end):
        self._start = start
        self._end = end
    
    def par_rate(self, curve):
        df_start = curve.get_df(self._start)
        df_end = curve.get_df(self._end)
        return -np.log(df_end / df_start) / (self._end - self._start)
        
if __name__ == '__main__':
    import curve
    grids = [1,2,3,4]
    dfs = [0.99, 0.98,0.97,0.96]
    c = curve.Curve(grids, dfs, interpolation_method = 'monotone_convex')
    start = 0
    end = 3
    simple = SimpleRate(start, end)    
    print('Simple rate :', simple.par_rate(c))  
    roll = 0.5
    swap = SingleCurrencySwap(start, end, roll)
    print('Swap rate :', swap.par_rate(c))
    
    
    