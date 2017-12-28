# -*- coding: utf-8 -*-
import numpy as np

def sabr_implied_volatility(forward, strike, tau, alpha, beta, rho, nu):
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
    


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    alpha = 0.1
    beta = 0.7
    rho = -0.2
    nu = 1
    fwd = 100
    tau = 1
    strikes = np.arange(0.7*fwd, 1.3*fwd)
    vols = sabr(fwd, strikes, tau, alpha, beta, rho, nu)
    plt.plot(strikes, vols, label = 'sabr')
    plt.legend()
    plt.show()
