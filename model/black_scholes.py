# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
import scipy.optimize

# Black76
def black_call(fwd, strike, volatility, tau):
    '''
    Calculate Call Option Price Under Black Model
    '''
    return fwd * ss.norm.cdf(_d1(fwd, strike, volatility, tau)) \
        - strike * ss.norm.cdf(_d2(fwd, strike, volatility, tau))
              
def black_put(fwd, strike, volatility, tau):
    '''
    Calculate Put Option Price Under Black Model
    '''
    return fwd - strike - black_call(fwd, strike, volatility, tau)

def implied_black_volatility(v, fwd, strike, tau, initial = 0.5):
    '''
    Calculate Implied Black Volatility
    '''
    return scipy.optimize.newton(
            lambda x : black_call(fwd, strike, x, tau) - v,
            initial)

def black_delta(fwd, strike, volatility, tau):
    '''
    Calculate Call Option Delta Under Black Model
    '''
    return ss.norm.cdf(_d1(fwd, strike, volatility, tau))

def strike_by_delta(fwd, delta, volatility, tau):
    '''
    Calculate Corresponding Strike Under Black Model
    '''
    return scipy.optimize.newton(
            lambda strike : black_delta(fwd, strike, volatility, tau) - delta - (1 if delta < 0 else 0),
            x0 = fwd)


def implied_black_volatilities(v, fwd, strike, tau, initial = 0.5):
    '''
    Calculate Implied Black Volatilities
    
    Arguments:
        v must be numpy.ndarray
    '''
    return scipy.optimize.fsolve(
            lambda x : black_call(fwd, strike, x, tau) - v,
            initial * np.ones(v.size))
    
def _d1(fwd, strike, volatility, tau):
    return (np.log(fwd/strike) +  volatility**2 / 2 * tau)/(volatility * np.sqrt(tau))
 
def _d2(fwd, strike, volatility, tau):
    return (np.log(fwd / strike)  - volatility**2 / 2 * tau) / (volatility * np.sqrt(tau))
 
    
# Normal
def normal_call(fwd, strike, volatility, tau):
    '''
    Calculate Call Option Price Under Normal Model
    '''
    return ((fwd - strike) * ss.norm.cdf(_d(fwd, strike, volatility, tau)) \
        + volatility * np.sqrt(tau) * ss.norm.pdf(_d(fwd, strike, volatility, tau)))

def normal_put(fwd, strike, volatility, tau):
    '''
    Calculate Put Option Price Under Normal Model
    '''
    return fwd - strike - normal_call(fwd, strike, volatility, tau)
   
def implied_normal_volatilities(v, fwd, strike, tau, initial = 0.002):
    return scipy.optimize.fsolve(
            lambda x : normal_call(fwd,strike,x,tau) - v,
            initial * np.ones(v.size))

def _d(fwd, strike, volatility, tau):
    return (fwd - strike) / (volatility * np.sqrt(tau))


# shifted
def shifted_black_call(fwd, strike, volatility, tau, shift):
    '''
    Calculate Call Option Price Under Shifted Log Normal Model
    '''
    return black_call(fwd + shift, strike + shift, volatility, tau)
    
def implied_shifted_volatilities(v, fwd, strike, tau, shift, initial = 0.1):
    return scipy.optimize.fsolve(
            lambda x : shifted_black_call(fwd, strike, x, tau, shift) - v,
            initial * np.ones(v.size))


if __name__ == '__main__':
    fwd = 100
    strike = 100
    vol = 0.1
    tau = 1
    v = black_call(fwd, strike, vol, tau)
    delta = -0.25
    deltastrike = strike_by_delta(fwd, delta, vol, tau)
    print('Strike :', deltastrike)
    print('Delta :',black_delta(fwd, deltastrike, vol, tau))
    
    print(implied_black_volatility(v, fwd, strike, tau))
    