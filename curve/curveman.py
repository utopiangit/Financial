# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
from functools import reduce
import curve

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
            lambda curve1, curve2 : curve.BasisCurve(curve1, curve2),
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


def _test2():
    import curve
#    import matplotlib.pyplot as plt
    grids1 = [1, 2, 3, 4]
    grid_df1 = [0.99, 0.985, 0.97, 0.965]
    curve1 = curve.Curve(grids1, grid_df1, 'log_linear')

    grids2 = [0.5, 1.5, 2.5 ,3.5, 4.5]
    grid_df2 = [0.99, 0.98, 0.965, 0.95, 0.94]
    curve2 = curve.Curve(grids2, grid_df2, 'monotone_convex')

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
    turn1_curve = curve.TurnCurve(turn1_from, turn1_to, turn1_size)
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


if __name__ == '__main__':
    _test2()
