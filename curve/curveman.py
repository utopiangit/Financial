# -*- coding: utf-8 -*-
import numpy as np

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


if __name__ == '__main__':
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
    