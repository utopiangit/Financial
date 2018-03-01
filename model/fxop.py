# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize

import black_scholes as bs

def market_strangle(fwd, atm, butterfly, delta, tau):
    call_strike = bs.strike_by_delta(fwd, delta, atm + butterfly, tau)
    call_premium = bs.black_call(fwd, call_strike, atm + butterfly, tau)
    put_strike = bs.strike_by_delta(fwd, -delta, atm + butterfly, tau)
    put_premium = bs.black_put(fwd, put_strike, atm + butterfly, tau)
    return call_premium + put_premium

def strike_by_delta_smile(fwd, delta, smile, tau):
    '''
    Arguments:
        smile : lambda fwd, strike, tau : vol

    '''
    return scipy.optimize.newton(
            lambda strike : bs.black_delta(fwd, strike, smile(fwd, strike, tau), tau) - delta - (1 if delta < 0 else 0),
            x0 = fwd)

def _test_smile():
    import sabr
    import matplotlib.pyplot as plt

    alpha = 0.5
    beta = 0.5
    rho = -0.2
    nu = 1
    smile = lambda fwd, strike, tau : sabr.sabr_implied_volatility(fwd, strike, tau, alpha, beta, rho, nu)

    fwd = 100
    delta = 0.1
    tau = 1
    strikes = np.arange(70, 130, 1)
    vols = smile(fwd, strikes, tau)
    print(vols)
    plt.plot(strikes, vols)
    plt.show()

    k = strike_by_delta_smile(fwd, delta, smile, tau)
    print(k)


def _test_ms():
    fwd = 100
    atm = 0.1
    bf = 0.01
    delta = 0.25
    tau = 1

    ms = market_strangle(fwd, atm, bf, delta, tau)
    print(ms)


if __name__ == '__main__':
    _test_smile()
