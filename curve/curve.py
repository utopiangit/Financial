# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as so
from collections import OrderedDict
from functools import reduce

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

    def update(self, discount_factors):
        self._discount_factors = discount_factors
        return self

    def get_degree_of_freedom(self):
        return len(self._grid_terms)

class BasisCurve(Curve):
    def __init__(self,
                 base_curve,
                 basis_curve):
#         super(BasisCurve, self).__init__(grid_terms,
#                                          discount_factors,
#                                          interpolation_method)
         self._base_curve = base_curve
         self._basis_curve = basis_curve

    def get_df(self, t):
        return self._base_curve.get_df(t) * self._basis_curve.get_df(t)

class TurnCurve(Curve):
    def __init__(self,
                 turn_from,
                 turn_to,
                 turn_size):
         self._turn_from = turn_from
         self._turn_to = turn_to
         self._turn_size = turn_size
         

    def get_df(self, t):
        df_turn_from = 1
        df_turn_to = np.exp(
            -self._turn_size * (self._turn_to - self._turn_from))
        df_inf = df_turn_to
        return interpolation.log_linear(
                t,
                np.array([self._turn_from, self._turn_to, 99]),
                np.array([df_turn_from, df_turn_to, df_inf])
            )

    def update(self, turn_size):
        self._turn_size = turn_size

    def get_degree_of_freedom(self):
        return 1

class CurveManager(object):
    '''CurveManager
    This class sotres some kinds of curves for multi curve framework.

    '''
    def __init__(self):
        self._curves = OrderedDict()
        self._basis_curves = {}

    def append_curve(self, curve_name, curve):
        self._curves[curve_name] = [
            curve,
            curve.get_degree_of_freedom()
        ]

    def register_basis_curve(self, curve_name, base_curve_names):
        curves = map(self.get_curve, base_curve_names)
        self._basis_curves[curve_name] = reduce(
            lambda curve1, curve2 : BasisCurve(curve1, curve2),
            curves)

    def get_curve(self, curve_name):
        if curve_name in self._curves:
            return self._curves[curve_name][0]
        else:
            return self._basis_curves[curve_name]

    def get_grids(self):
        return [(curve_name, content[1]) for (curve_name, content)
                in self._curves.items()]

    def update_curves(self, values):
        counter = 0
        for content in self._curves.values():
            curve = content[0]
            num_grids = content[1]
            curve.update(values[counter:counter + num_grids])
            counter = counter + num_grids
        return self

class CurveEngine(object):
    def __init__(self,
                 curve_manager,
                 instruments,
                 loss_function):
        self.__curve_manager = curve_manager
        self.__instruments = instruments
        self.__loss_function = loss_function

    def build_curve(self, market_rates):
        def loss(parameters):
            cm = self.__curve_manager
            cm.update_curves(parameters)
            evaluated = np.array([
                instrument.par_rate(cm) for instrument in self.__instruments])
            return self.__loss_function(market_rates, evaluated)
        num_freedom = np.sum([grid[1] for grid in self.__curve_manager.get_grids()])
        param = so.minimize(loss,
                            np.ones((num_freedom, )),
                            tol = 1e-8,
                            method = 'nelder-mead')
        print('param:', param.x)
        return self.__curve_manager.update_curves(param.x)


def build_curve(curve, instruments, market_rates):
    def loss(dfs):
        par_rates = np.array(
            [ins.par_rate(curve.update(dfs)) for ins in instruments]).flatten()
        return np.sum(np.power((par_rates - market_rates), 2))
    opt = so.minimize(loss,
                      curve._discount_factors,
                      method = 'Nelder-Mead',
                      tol = 1e-6)
    curve.update(opt.x)
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
    loss = lambda x: np.sum(np.power(insts.par_rate(mc.update(x)) - swap_rates, 2))
#    print('before :', loss(dfs))
    t0 = time.time()
    param = so.minimize(loss,
                        dfs,
                        tol = 1e-8)
    t1 = time.time()
#    print('after:', loss(param.x))
    print('vectorized :', t1 - t0, 's')

    mc.update(param.x)
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

    basis_grids = np.array([1, 2, 3, 4, 5])
    basis_rates = np.array([0.00, 0.001, 0.001, 0.002, 0.003])
    basis_dfs = np.exp(-np.array(basis_rates) * basis_grids)
    print('basis_df :',basis_dfs)
    basis_curve = Curve(basis_grids, basis_dfs, 'monotone_convex')

    composite_curve = BasisCurve(base_curve, basis_curve)

    ts = np.arange(0, 5, 1/365)
    df_base = base_curve.get_df(ts)
    df_basis = basis_curve.get_df(ts)
    df_composite = composite_curve.get_df(ts)
    plt.plot(ts, df_base, label = 'base')
    plt.plot(ts, df_basis, label = 'basis')
    plt.plot(ts, df_composite, label = 'composite')
    plt.legend()
    plt.show()

    fwd_base = -np.log(df_base[1:] / df_base[:-1]) / (ts[1:] - ts[:-1])
    fwd_basis = -np.log(df_basis[1:] / df_basis[:-1]) / (ts[1:] - ts[:-1])
    fwd_composite = -np.log(df_composite[1:] / df_composite[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd_base, label = 'base')
    plt.plot(ts[:-1], fwd_basis, label = 'basis')
    plt.plot(ts[:-1], fwd_composite, label = 'composite')
    plt.legend()
    plt.show()

def _test_turn():
    import matplotlib.pyplot as plt
    print('The turn curve')

    base_grids = np.array([0.5, 1, 1.5, 2, 2.5])
    base_rates = np.array([0.005, 0.004, 0.006, 0.008, 0.01])

    base_dfs = np.exp(-np.array(base_rates) * base_grids)
    base_curve = Curve(base_grids, base_dfs)

    turn1_from = 0.25
    turn1_to = 0.3
    turn1_size = 0.01
    turn1_curve = TurnCurve(turn1_from, turn1_to, turn1_size)

    turn2_from = 0.7
    turn2_to = 0.75
    turn2_size = 0.005
    turn2_curve = TurnCurve(turn2_from, turn2_to, turn2_size)

    turned_curve = BasisCurve(base_curve, turn1_curve)
    turned_curve = BasisCurve(turned_curve, turn2_curve)

    ts = np.arange(0, 3, 1/365)
    df_base = base_curve.get_df(ts)
    df_turned = turned_curve.get_df(ts)
    plt.plot(ts, df_base, label = 'base')
    plt.plot(ts, df_turned, label = 'turned')
    plt.legend()
    plt.show()

    fwd_base = -np.log(df_base[1:] / df_base[:-1]) / (ts[1:] - ts[:-1])
    fwd_turned = -np.log(df_turned[1:] / df_turned[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd_base, label = 'base')
    plt.plot(ts[:-1], fwd_turned, label = 'turned')
    plt.plot(ts[:-1], fwd_turned - fwd_base, label = 'diff')
    plt.legend()
    plt.show()

def _test_curveman():
#    import matplotlib.pyplot as plt
    grids1 = [1, 2, 3, 4]
    grid_df1 = [0.99, 0.985, 0.97, 0.965]
    curve1 = Curve(grids1, grid_df1, 'log_linear')

    grids2 = [0.5, 1.5, 2.5 ,3.5, 4.5]
    grid_df2 = [0.99, 0.98, 0.965, 0.95, 0.94]
    curve2 = Curve(grids2, grid_df2, 'monotone_convex')

    cm = CurveManager()
    cm.append_curve('JPY-OIS',
                    curve1)
    cm.append_curve('JPY-LO',
                    curve2)

    print(cm.get_grids())
    print('JPYOIS', cm.get_curve('JPY-OIS').get_df(2.4))
    print('JPYLO', cm.get_curve('JPY-LO').get_df(2.4))

    cm.register_basis_curve('JPY-LIBOR',
                            ['JPY-OIS', 'JPY-LO'])

    print('JPYLIBOR', cm.get_curve('JPY-LIBOR').get_df(2.4))

    new_values = [0.99, 0.985,0.97,0.965, 1, 0.99, 0.985, 0.97, 0.96]
    cm.update_curves(new_values)
    print('-----updated-----')
    print('JPYOIS', cm.get_curve('JPY-OIS').get_df(2.4))
    print('JPYLO', cm.get_curve('JPY-LO').get_df(2.4))
    print('JPYLIBOR', cm.get_curve('JPY-LIBOR').get_df(2.4))

    turn1_from = 0.25
    turn1_to = 0.3
    turn1_size = 0.01
    turn1_curve = TurnCurve(turn1_from, turn1_to, turn1_size)
    cm.append_curve('Turn1',
                    turn1_curve)
    cm.register_basis_curve('JPY-LIBOR2',
                            ['JPY-OIS', 'JPY-LO', 'Turn1'])
    print(cm.get_grids())                            
    print('Turned JPYLIBOR', cm.get_curve('JPY-LIBOR2').get_df(2.4))

    import matplotlib.pyplot as plt
    ts = np.arange(0, 3, 1/365)
    df_ois = cm.get_curve('JPY-OIS').get_df(ts)
    df_lo = cm.get_curve('JPY-LO').get_df(ts)
    df = cm.get_curve('JPY-LIBOR2').get_df(ts)

    fwd_ois = -np.log(df_ois[1:] / df_ois[:-1]) / (ts[1:] - ts[:-1])
    fwd_lo = -np.log(df_lo[1:] / df_lo[:-1]) / (ts[1:] - ts[:-1])
    fwd = -np.log(df[1:] / df[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd_ois, label = 'ois')
    plt.plot(ts[:-1], fwd_lo, label = 'lo')
    plt.plot(ts[:-1], fwd, label = 'turned')
    plt.legend()
    plt.show()

def _test_curveng():
    import cminstruments as inst
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # Define instruments which we calibrate curves to
    grids_fixed_ois = [7/365, 1/12, 2/12, 3/12, 6/12, 9/12, 1]
    fixed_ois = [inst.ZeroRate(0, end, 'JPY-OIS') for end in grids_fixed_ois]
    fixed_ois_rates = np.array([0.001, 0.001, 0.0015, 0.002, 0.0022, 0.003, 0.0035])

    start_fra = [0, 3/12, 6/12]
    end_fra = [6/12, 9/12, 1]
    fra = [inst.SimpleRate(start, end, 'JPY-LIBOR') for start, end in zip(start_fra ,end_fra)]
    fra_rates = [0.004, 0.007, 0.01]

    # Create CurveManager
    cm = CurveManager()

    # Define curves and register them to CurveManager
    grids_ois = [7/365, 1/12, 0.4, 0.6, 1]
    dfs_ois = np.ones((len(grids_ois), ))
    ois_base = Curve(grids_ois, dfs_ois, 'log_linear')
    cm.append_curve('JPY-OIS-BASE', ois_base)

    turn1_from = 0.2
    turn1_to = 0.21
    turn1_size = 0.0
    turn1 = TurnCurve(turn1_from, turn1_to, turn1_size)
    cm.append_curve('JPY-Turn1', turn1)

    turn2_from = 0.7
    turn2_to = 0.71
    turn2_size = 0.0
    turn2 = TurnCurve(turn2_from, turn2_to, turn2_size)
    cm.append_curve('JPY-Turn2', turn2)

    cm.register_basis_curve('JPY-OIS',
                            ['JPY-OIS-BASE', 'JPY-Turn1', 'JPY-Turn2'])

    grids_lo = end_fra
    dfs_lo = np.ones((len(grids_lo), ))
    lo = Curve(grids_lo, dfs_lo, 'monotone_convex')
    cm.append_curve('JPY-LO', lo)

    cm.register_basis_curve('JPY-LIBOR',
                            ['JPY-OIS', 'JPY-LO'])

    print(cm.get_grids())

    loss = lambda xs, ys: np.sum((np.array(xs) - np.array(ys))**2)
    ce = CurveEngine(cm,
                     fixed_ois + fra,
                     loss)
    t0 = time.time()
    cm = ce.build_curve(np.concatenate((fixed_ois_rates, fra_rates)))
    t1 = time.time()
    print('time :', t1 - t0)
    ois_evaluated = np.array([ois.par_rate(cm) for ois in fixed_ois])
    fra_evaluated = np.array([inst.par_rate(cm) for inst in fra])
    print('calibrated ois :', ois_evaluated)
    print('calibrated fra :', fra_evaluated)

    ts = np.arange(0, 1, 1/365)
    df_ois = cm.get_curve('JPY-OIS').get_df(ts)
    df_libor = cm.get_curve('JPY-LIBOR').get_df(ts)

    fwd_ois = -np.log(df_ois[1:] / df_ois[:-1]) / (ts[1:] - ts[:-1])
    fwd_libor = -np.log(df_libor[1:] / df_libor[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd_ois, label = 'ois')
    plt.plot(ts[:-1], fwd_libor, label = 'libor')
    plt.plot(ts[:-1], fwd_libor - fwd_ois, label = 'lo')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    #_test_build_curve()
#    _vectorized_calib()
#    _test_basis()
    _test_turn()
