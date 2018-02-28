# -*- coding: utf-8 -*-
import numpy as np
import black_scholes as bs
import scipy.optimize as so

def sabr_implied_volatility(forward, strike, tau, alpha, beta, rho, nu):
    '''
    Calculate Implied Volatility Under SABR Model by Hagan et al(2002).

    '''
    x = np.log(forward / strike)
    fkbeta = (forward * strike)**((1. - beta) / 2.0)
    zeta = nu / alpha * fkbeta * x
    chi = np.log(((1. - 2. * rho * zeta + zeta**2)**0.5 + zeta - rho) / (1. - rho))
    A = 1. + ((1. - beta)**2 / 24. * x**2 + (1. - beta)**4 / 1920. * x**4)
    B = (1. - beta)**2 / 24. * alpha**2 / (fkbeta ** 2)
    C = alpha * beta * rho * nu / (4. * fkbeta)
    D = (2. - 3. * rho**2) * nu**2 / 24.
    E = 1. + (B + C + D) * tau
    return np.where(np.isclose(chi,  0),
                    alpha / fkbeta * E,
                    alpha / (fkbeta * A) * zeta / chi * E)

def strike_by_delta(forward, delta, tau, alpha, beta, rho, nu):
    '''
    Calculate Corresponding Strike Under SABR Model
    '''
    error = lambda strike : bs.black_delta(
        forward, strike, sabr_implied_volatility(forward, strike, tau, alpha, beta, rho, nu), tau) - delta - (1 if delta < 0 else 0)
    return so.newton(error, x0 = fwd)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    alpha = 0.3
    beta = 0.7
    rho = -0.2
    nu = 1
    fwd = 100
    tau = 1
    strikes = np.arange(0.7*fwd, 1.3*fwd)
    vols = sabr_implied_volatility(fwd, strikes, tau, alpha, beta, rho, nu)
    plt.plot(strikes, vols, label = 'sabr')
    plt.legend()
    plt.show()

    delta = -0.25
    deltastrike = strike_by_delta(fwd, delta, tau, alpha, beta, rho, nu)
    print('Strike :', deltastrike)
