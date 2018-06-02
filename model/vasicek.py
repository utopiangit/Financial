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
        self.__volatility = volatility
        self.__mean_level = mean_level
        self.__mean_reversion = mean_reversion
        self.__t_initial = t_initial

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
        return ((self.__mean_level - self.__volatility**2 / (2 * self.__mean_reversion**2)) * (B - t_maturity + t_observe)
            - self.__volatility**2 / (4 * self.__mean_reversion) * B**2)

    def func_b(self,
               t_obesrve: float, 
               t_maturity: float) -> float:
        return (1 - np.exp(-self.__mean_reversion * (t_maturity - t_obesrve))) / self.__mean_reversion

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    kappa = 0.1717
    theta = 0.06790
    volatility = 0.0009
    r0 = 0.002
    model = Vasicek(volatility, theta, kappa)
    curve = model.get_curve(0, r0)
    ts = np.arange(0, 5, 5. / 365)
    df = curve.get_df(ts)
    plt.plot(ts, df)
    plt.show()

    t_libors = np.array([1/12, 2/12, 3/12, 6/12, 1])
    libors = 1 / t_libors * (1 / curve.get_df(t_libors) - 1)
    t_swaps = np.array([2, 3, 5, 7, 10, 15, 30])
    swaps = -np.log(curve.get_df(t_swaps)) / t_swaps
    print(libors)
    print(swaps)