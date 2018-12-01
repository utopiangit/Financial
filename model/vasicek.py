# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable
import curve_wrapper as cw

class Vasicek(object):
    def __init__(self, 
                 volatility: float, 
                 mean_level: float, 
                 mean_reversion: float, 
                 t_initial: float = 0):
        self.volatility = volatility
        self.mean_level = mean_level
        self.mean_reversion = mean_reversion
        self.t_initial = t_initial

    def get_curve(self, 
                  t_observe: float, 
                  r_observe: float):
        return cw.CurveWrppaer(
            lambda t_maturity : np.exp(self.func_a(t_observe, t_maturity) 
                - self.func_b(t_observe, t_maturity) * r_observe))

    def func_a(self, 
               t_observe: float, 
               t_maturity: float) -> float:
        B = self.func_b(t_observe, t_maturity)
        return ((self.mean_level - self.volatility**2 / (2 * self.mean_reversion**2)) * (B - t_maturity + t_observe)
            - self.volatility**2 / (4 * self.mean_reversion) * B**2)

    def func_b(self,
               t_obesrve: float, 
               t_maturity: float) -> float:
        return (1 - np.exp(-self.mean_reversion * (t_maturity - t_obesrve))) / self.mean_reversion

    def proceed(self, 
                r,
                timestep,
                rnd_normal):
        return (r + self.mean_reversion * (self.mean_level - r) * timestep 
            + self.volatility * np.sqrt(timestep) * rnd_normal)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    kappa = 0.1717
    theta = 0.06790
    volatility = 0.0009
    r0 = 0.002
    model = Vasicek(volatility, theta, kappa)
    curve1 = model.get_curve(0, r0)
    curve2 = model.get_curve(1, r0)

    ts = np.arange(0, 5, 5. / 365)
    df1 = curve1.get_df(ts)
    df2 = curve2.get_df(ts)
    plt.plot(ts, df1, label = 'original')
    plt.plot(ts, df2, label = 'shocked')
    plt.legend()
    plt.show()

    t_libors = np.array([1/12, 2/12, 3/12, 6/12, 1])
    libors = 1 / t_libors * (1 / curve1.get_df(t_libors) - 1)
    t_swaps = np.array([2, 3, 5, 7, 10, 15, 30])
    swaps = -np.log(curve1.get_df(t_swaps)) / t_swaps
    print(libors)
    print(swaps)
