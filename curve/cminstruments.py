# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np

class Instrument(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def par_rate(self, curve_manager):
        pass

class SimpleRate(Instrument):
    def __init__(self, start, end, curve_name):
        self._start = start
        self._end = end
        self._curve_name = curve_name

    def par_rate(self, curve_manager):
        curve = curve_manager.get_curve(self._curve_name)
        df_start = curve.get_df(self._start)
        df_end = curve.get_df(self._end)
        return (df_start / df_end - 1) / (self._end - self._start)


class FloatFixedSwap(Instrument):
    def __init__(self, 
                 start, 
                 end, 
                 roll,
                 name_float_curve,
                 name_discount_curve):
        self._start = start
        self._end = end
        self._roll = roll
        self._name_float_curve = name_float_curve
        self._name_discount_curve = name_discount_curve

    def _generate_payment_dates(self, start, end, roll):
        # simple implementation for ease
        return np.arange(start, end, roll) + roll

    def par_rate(self, curve_manager):
        float_curve = curve_manager.get_curve(self._name_float_curve)
        discount_curve = curve_manager.get_curve(self._name_discount_curve)
        df_start = float_curve.get_df(self._start)
        df_end = float_curve.get_df(self._end)
        payment_dates = self._generate_payment_dates(self._start,
                                                     self._end,
                                                     self._roll)
        annuity = np.sum(discount_curve.get_df(payment_dates)) * self._roll
        return (df_start - df_end) / annuity

class ZeroRate(Instrument):
    def __init__(self, start, end, curve_name):
        self._start = start
        self._end = end
        self._curve_name = curve_name

    def par_rate(self, curve_manager):
        curve = curve_manager.get_curve(self._curve_name)
        df_start = curve.get_df(self._start)
        df_end = curve.get_df(self._end)
        return -np.log(df_end / df_start) / (self._end - self._start)


def __test_curveman():
    import curve
    import curveman
    import numpy as np
    import scipy.optimize as so
    import matplotlib.pyplot as plt

    # Define instruments which we calibrate curves to
    grids_fixed_ois = [7/365, 1/12, 2/12, 3/12, 6/12, 9/12, 1]
    fixed_ois = [ZeroRate(0, end, 'JPY-OIS') for end in grids_fixed_ois]
    mid_fixed_ois = np.array([0.001, 0.001, 0.0015, 0.002, 0.0022, 0.003, 0.0035])

    # Create CurveManager
    cm = curveman.CurveManager()

    # Define curves and register them to CurveManager
    grids_ois = [7/365, 0.2, 0.4, 0.6, 1]
    dfs_ois = np.ones((len(grids_ois), ))
    ois_base = curve.Curve(grids_ois, dfs_ois, 'log_linear')
    cm.append_curve('JPY-OIS-BASE', ois_base)

    turn1_from = 0.25
    turn1_to = 0.26
    turn1_size = 0.01
    turn1 = curve.TurnCurve(turn1_from, turn1_to, turn1_size)
    cm.append_curve('JPY-Turn1', turn1)

    turn2_from = 0.75
    turn2_to = 0.76
    turn2_size = 0.02
    turn2 = curve.TurnCurve(turn2_from, turn2_to, turn2_size)
    cm.append_curve('JPY-Turn2', turn2)

    cm.register_basis_curve('JPY-OIS',
                            ['JPY-OIS-BASE', 'JPY-Turn1', 'JPY-Turn2'])

    print(cm.get_grids())

    def loss(dfs):
        cm.update_curves(dfs)
        mids = np.array([ois.par_rate(cm) for ois in fixed_ois])
        return np.sum((mid_fixed_ois - mids)**2)
    #mids = [ois.par_rate(cm) for ois in fixed_ois]
    initial = np.ones((len(grids_fixed_ois), ))
    print('initial loss :', loss(initial))

    param = so.minimize(loss,
                        initial,
                        tol = 1e-6)
    print('after calib :', loss(param.x))
    print('df :', param.x)

    cm.update_curves(param.x)
    ts = np.arange(0, 1, 1/365)
    df = cm.get_curve('JPY-OIS').get_df(ts)

    fwd = -np.log(df[1:] / df[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd, label = 'turned')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    __test_curveman()