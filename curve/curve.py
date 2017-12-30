# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as so
import interpolation

class Curve(object):
    def __init__(self, 
                 grid_terms,
                 discount_factors,
                 interpolation_method = 'log_linear'):
        self._grid_terms = grid_terms
        self._discount_factors = discount_factors
        self._interpolation_method = interpolation_method
    
    def get_df(self, t):
        switcher = {
            'log_linear' : interpolation.log_linear,
            'log_cubic' : interpolation.log_cubic,
            'monotone_convex' : interpolation.monotone_convex
            }
        return switcher[self._interpolation_method](t,
                                                    self._grid_terms,
                                                    self._discount_factors)
    
    def _update(self, discount_factors):
        self._discount_factors = discount_factors
        return self
            
class BasisCurve(Curve):
    def __init__(self,
                 base_curve,
                 grid_terms,
                 discount_factors,
                 interpolation_method = 'log_linear'):
         super(BasisCurve, self).__init__(grid_terms,
                                          discount_factors,
                                          interpolation_method)
         self._base_curve = base_curve
#         self._grid_terms = grid_terms
#         self._discount_factors = discount_factors
#         self._interpolation_method = interpolation_method
    
    def get_df(self, t):
        return self._base_curve.get_df(t) * super().get_df(t)


def build_curve(curve, instruments, market_rates):
    def loss(dfs):        
        par_rates = np.array(
            [ins.par_rate(curve._update(dfs)) for ins in instruments]).flatten()
        return np.sum(np.power((par_rates - market_rates), 2))
    opt = so.minimize(loss,
                      curve._discount_factors,
                      method = 'Nelder-Mead',
                      tol = 1e-6)
    curve._update(opt.x)
    return curve

def _test_build_curve():
    import instruments as inst    
    import matplotlib.pyplot as plt

    print('===== Building curves whose swap rates are the same =====')
    # market data to fit
    start_dates = [0, 0, 0, 0, 0]
    end_dates = [1, 2, 3, 4, 5]
    roll = 0.5
    swap_rates = [0.01, 0.011, 0.012, 0.015, 0.016]
    instruments = [inst.SingleCurrencySwap(start, end, roll) 
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

    print('log linear :', [inst.par_rate(built_linear) for inst in instruments])    
    print('log cubic :', [inst.par_rate(built_cubic) for inst in instruments])
    print('monotone convex :', [inst.par_rate(built_mc) for inst in instruments])

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

def _vectorized_calib():
    import instruments as inst    
    import time
    print('===== Compare the time to optimize curves =====')    
    
    start_dates = np.array([0, 0, 0, 0, 0])
    end_dates = np.array([1, 2, 3, 4, 5])
    swap_rates = np.array([0.01, 0.011, 0.013, 0.015, 0.016])
    insts = inst.SimpleRate(start_dates, end_dates)

    grids = np.array(end_dates)
    dfs = np.exp(-np.array(swap_rates) * grids)
    mc = Curve(grids, dfs, interpolation_method = 'monotone_convex')
    loss = lambda x: np.sum(np.power(insts.par_rate(mc._update(x)) - swap_rates, 2))
#    print('before :', loss(dfs))
    t0 = time.time()
    param = so.minimize(loss, 
                        dfs,
                        tol = 1e-8)
    t1 = time.time()
#    print('after:', loss(param.x))
    print('vectorized :', t1 - t0, 's')
        
    mc._update(param.x)
    print(insts.par_rate(mc))
#    print(param)


    insts = [inst.SimpleRate(start, end) for start, end in zip(start_dates, end_dates)]
    mc2 = Curve(grids, dfs, interpolation_method = 'monotone_convex')
    t2 = time.time()
    build_curve(mc2, insts, swap_rates)    
    t3 = time.time()
    print('build :', t3 - t2, 's')
    print([inst.par_rate(mc2) for inst in insts])

    
def _test_basis():
    import matplotlib.pyplot as plt
    print('Plot of a base curve and a basis curve.')
    print('The basis curve is built on the base curve.')

    base_grids = np.array([0.5, 1, 1.5, 2, 2.5])
    base_rates = np.array([0.005, 0.004, 0.006, 0.008, 0.01])

    base_dfs = np.exp(-np.array(base_rates) * base_grids)
    base_curve = Curve(base_grids, base_dfs)

    basis_grids = np.array([1,2,3,4,5])
    basis_rates = np.array([0.004, 0.006, 0.006, 0.007, 0.008])    
    basis_dfs = np.exp(-np.array(basis_rates) * basis_grids)
    basis_curve = BasisCurve(base_curve, basis_grids, basis_dfs, 'monotone_convex')
    
    ts = np.arange(0, 5, 1/365)
    df_base = base_curve.get_df(ts)
    df_basis = basis_curve.get_df(ts)
    plt.plot(ts, df_base, label = 'base')
    plt.plot(ts, df_basis, label = 'basis')
    plt.legend()
    plt.show()    

    fwd_base = -np.log(df_base[1:] / df_base[:-1]) / (ts[1:] - ts[:-1])
    fwd_basis = -np.log(df_basis[1:] / df_basis[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd_base, label = 'base')
    plt.plot(ts[:-1], fwd_basis, label = 'basis')    
    plt.legend()
    plt.show()    

    
if __name__ == '__main__':
#    _test_build_curve()
#    _vectorized_calib()
    _test_basis()