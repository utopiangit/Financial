# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as so
import interpolation

class Curve(object):
    def __init__(self, 
                 grid_terms,
                 discount_factors,
                 interpolation_method = 'log_linear'):
        self._grid_terms = np.array(grid_terms)
        self._discount_factors = np.array(discount_factors)
        self._interpolation_method = interpolation_method 
    
    def get_df(self, t):
        switcher = {
            'log_linear' : 
                interpolation.log_linear(t,
                                         self._grid_terms,
                                         self._discount_factors),
            'log_cubic' : 
                interpolation.log_cubic(t,
                                        self._grid_terms,
                                        self._discount_factors),

            'monotone_convex' : 
                interpolation.monotone_convex(t,
                                              self._grid_terms,
                                              self._discount_factors)
            }
        return switcher[self._interpolation_method]
    
    def _update(self, discout_dactors):
        self._discount_factors = discout_dactors
            

def build_curve(curve, instruments, market_rates):
    def loss(dfs):
        curve._update(dfs)
        par_rates = np.array([ins.par_rate(curve) for ins in instruments])
        return np.sum(np.power((par_rates - market_rates), 2))
    opt = so.minimize(loss,
                      curve._discount_factors,
                      method = 'CG',
                      tol = 1e-6)
    curve._update(opt.x)
    return curve

def _test_build_curve():
    import instruments as inst    
    import matplotlib.pyplot as plt
    import pickle

    # market data to fit
    start_dates = [0, 0, 0, 0, 0]
    end_dates = [1, 2, 3, 4, 5]
    roll = 0.5
    swap_rates = [0.01, 0.012, 0.012, 0.015, 0.016]
    instruments = [inst.Single_currency_swap(start, end, roll) 
        for start, end in zip(start_dates, end_dates)]

    # generate curve grids same as swap end dates
    grids = np.array(end_dates)
    dfs = np.exp(-np.array(swap_rates) * grids)
    linear = Curve(grids, dfs, interpolation_method = 'log_linear')
    cubic = Curve(grids, dfs, interpolation_method = 'log_cubic')
    mc = Curve(grids, dfs, interpolation_method = 'monotone_convex')
    

    built_linear = build_curve(linear, instruments, swap_rates)
    built_cubic = build_curve(cubic, instruments, swap_rates)
    built_mc = build_curve(mc, instruments, swap_rates)
    ts = np.arange(0, 5, 1. / 365)
    df_linear = built_linear.get_df(ts)
    df_cubic = built_cubic.get_df(ts)
    df_mc = built_mc.get_df(ts)
    plt.plot(ts, df_linear, label = 'log linear')
    plt.plot(ts, df_cubic, label = 'log cubic')
    plt.plot(ts, df_mc, label = 'monotone convex')
    plt.legend()    
    plt.show()
    
    fwd_linear = -np.log(df_linear[1:] / df_linear[:-1]) / (ts[1:] - ts[:-1])
    fwd_cubic = -np.log(df_cubic[1:] / df_cubic[:-1]) / (ts[1:] - ts[:-1])
    fwd_mc = -np.log(df_mc[1:] / df_mc[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd_linear, label = 'log linear')
    plt.plot(ts[:-1], fwd_cubic, label = 'log cubic')
    plt.plot(ts[:-1], fwd_mc, label = 'monotone convex')
    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100))
    plt.show()
    
    print('mc grid :',built_mc._grid_terms)
    print('mc df :',built_mc._discount_factors)
    with open('curve.pickle', mode='wb') as f:
        pickle.dump(built_mc, f)
    
if __name__ == '__main__':
    _test_build_curve()
    