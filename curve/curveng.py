# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as so

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

def __test():
    import curve
    import curveman
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
    cm = curveman.CurveManager()

    # Define curves and register them to CurveManager
    grids_ois = [7/365, 1/12, 0.4, 0.6, 1]
    dfs_ois = np.ones((len(grids_ois), ))
    ois_base = curve.Curve(grids_ois, dfs_ois, 'log_linear')
    cm.append_curve('JPY-OIS-BASE', ois_base)

    turn1_from = 0.2
    turn1_to = 0.21
    turn1_size = 0.0
    turn1 = curve.TurnCurve(turn1_from, turn1_to, turn1_size)
    cm.append_curve('JPY-Turn1', turn1)

    turn2_from = 0.7
    turn2_to = 0.71
    turn2_size = 0.0
    turn2 = curve.TurnCurve(turn2_from, turn2_to, turn2_size)
    cm.append_curve('JPY-Turn2', turn2)

    cm.register_basis_curve('JPY-OIS',
                            ['JPY-OIS-BASE', 'JPY-Turn1', 'JPY-Turn2'])

    grids_lo = end_fra
    dfs_lo = np.ones((len(grids_lo), ))
    lo = curve.Curve(grids_lo, dfs_lo, 'monotone_convex')
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
    
def __test_mc():
    import curve
    import matplotlib.pyplot as plt
    import interpolation as interp
    result = [0.99998082, 0.99991667, 0.9992836,  0.99868419, 0.99668883, 0.00833426, 
              0.00999793, 0.99910109, 0.9979804, 0.99800533]
    dfs = result[-3:]
    grids = [6/12, 9/12, 1]
    #curve = curve.Curve(grids, dfs, 'monotone_convex')

    ts = np.arange(0, 1, 1/365)
    #df = curve.get_df(ts)
    df = interp.monotone_convex(ts, grids, dfs)
    #df = interp.log_cubic(ts, grids, dfs)
    fwd = -np.log(df[1:] / df[:-1]) / (ts[1:] - ts[:-1])
    #plt.plot(ts, df)
    plt.plot(ts[:-1], fwd)
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    print('curveng')
    __test()
    #__test_mc()