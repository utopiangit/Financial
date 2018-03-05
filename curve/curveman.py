# -*- coding: utf-8 -*-
import numpy as np

from collections import OrderedDict

class CurveMan(object):
    def __init__(self):
        self._curves = OrderedDict()

    def add_curve(self, curve_name, curve, num_grids):
        self._curves[curve_name] = [
            curve,
            num_grids
        ]

    def get_curve(self, curve_name):
        return self._curves[curve_name][0]

    def get_grids(self):
        return [(curve_name, content[1]) for (curve_name, content)
                in self._curves.items()]

    def update_curves(self, values):
        counter = 0
        for content in self._curves.values():
            curve = content[0]
            num_grids = content[1]
            curve.update(values[counter:counter + num_grids])
            print(values[counter:counter + num_grids])
            counter = counter + num_grids



class CurveManager(object):
    '''CurveManager
    This class sotres some kinds of curves for multi curve framework.

    Kinds of curves:
        'regular' -- Regular tenor curve of the currency.
        'xxx-ois' -- Curve collateralized with currency 'xxx'.
        '3m', '6m' etc. -- Curve corresponding to the tenor.

    '''
    REGULAR = 'regular'
    OIS = '-ois'

    def __init__(self):
        self._dict = {}


    def add_curve(self, curve, currency, kind = REGULAR):
        if not(currency in self._dict):
            self._dict[currency] = {}
        self._dict[currency][kind] = curve

    def add_collateralized_curve(self, curve, currency, collateral_currency):
        self.add_curve(curve, currency, collateral_currency + self.OIS)

    def __call__(self, currency, *, kind = None, collateral_currency = None):
        if (kind is None) and (collateral_currency is None):
            kind = self.REGULAR
        elif kind is None:
            kind = collateral_currency + self.OIS
        return self._dict[currency][kind]


def _test1():
    import curve
    import matplotlib.pyplot as plt
    grids = [1,2,3,4]
    grid_df = [0.99, 0.985,0.97,0.965]
    curve1 = curve.Curve(grids, grid_df, 'log_linear')
    cm = CurveManager()
    ccy1 = 'JPY'
    kind1 = 'regular'
    cm.add_curve(ccy1, kind1, curve1)

    ts = np.arange(0, 4, 1/365)
    df = cm(ccy1, kind1).get_df(ts)
    plt.plot(ts, df)
    plt.show()

def _test2():
    import curve
    import matplotlib.pyplot as plt
    grids1 = [1,2,3,4]
    grid_df1 = [0.99, 0.985,0.97,0.965]
    curve1 = curve.Curve(grids1, grid_df1, 'log_linear')

    grids2 = [0.5, 1.5, 2.5 ,3.5, 4.5]
    grid_df2 = [0.99, 0.98, 0.965, 0.95, 0.94]
    curve2 = curve.Curve(grids2, grid_df2, 'log_linear')

    cm = CurveMan()
    cm.add_curve('JPY-OIS',
                 curve1,
                 len(grids1))
    cm.add_curve('JPY-6M',
                 curve2,
                 len(grids2))

    print(cm.get_grids())
    print(cm.get_curve('JPY-6M').get_df(2.4))

    new_values = [0.99, 0.985,0.97,0.965, 1, 0.99, 0.985, 0.97, 0.96]
    cm.update_curves(new_values)
    print(cm.get_curve('JPY-6M').get_df(1))


if __name__ == '__main__':
    _test2()
