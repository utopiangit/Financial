# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
import scipy.optimize

# Black76
def black_call(fwd, strike, sigma, tau):
    '''
    Calculate Call Option Price Under Black Model
    '''
    return fwd * ss.norm.cdf(_d1(fwd, strike, sigma, tau)) \
        - strike * ss.norm.cdf(_d2(fwd, strike, sigma, tau))
              
def black_put(fwd, strike, sigma, tau):
    '''
    Calculate Put Option Price Under Black Model
    '''
    return fwd - strike - black_call(fwd, strike, sigma, tau)

def implied_black_volatility(v, fwd, strike, tau, initial = 0.5):
    '''
    Calculate Implied Black Volatility
    '''
    return scipy.optimize.newton(
            lambda x : black_call(fwd, strike, x, tau) - v,
            initial)

def implied_black_volatilities(v, fwd, strike, tau, initial = 0.5):
    '''
    Calculate Implied Black Volatilities
    
    Arguments:
        v must be numpy.ndarray
    '''
    return scipy.optimize.fsolve(
            lambda x : black_call(fwd, strike, x, tau) - v,
            initial * np.ones(v.size))
    
def _d1(fwd, strike, sigma, tau):
    return (np.log(fwd/strike) +  sigma**2 / 2 * tau)/(sigma * np.sqrt(tau))
 
def _d2(fwd, strike, sigma, tau):
    return (np.log(fwd / strike)  - sigma**2 / 2 * tau) / (sigma * np.sqrt(tau))
 
    
# Normal
def normal_call(fwd, strike, sigma, tau):
    '''
    Calculate Call Option Price Under Normal Model
    '''
    return ((fwd - strike) * ss.norm.cdf(_d(fwd, strike, sigma, tau)) \
        + sigma * np.sqrt(tau) * ss.norm.pdf(_d(fwd, strike, sigma, tau)))

def normal_put(fwd, strike, sigma, tau):
    '''
    Calculate Put Option Price Under Normal Model
    '''
    return fwd - strike - normal_call(fwd, strike, sigma, tau)
   
def implied_normal_volatilities(v, fwd, strike, tau, initial = 0.002):
    return scipy.optimize.fsolve(
            lambda x : normal_call(fwd,strike,x,tau) - v,
            initial * np.ones(v.size))

def _d(fwd, strike, sigma, tau):
    return (fwd - strike) / (sigma * np.sqrt(tau))


# shifted
def shifted_black_call(fwd, strike, sigma, tau, shift):
    '''
    Calculate Call Option Price Under Shifted Log Normal Model
    '''
    return black_call(fwd + shift, strike + shift, sigma, tau)
    
def implied_shifted_volatilities(v, fwd, strike, tau, shift, initial = 0.1):
    return scipy.optimize.fsolve(
            lambda x : shifted_black_call(fwd, strike, x, tau, shift) - v,
            initial * np.ones(v.size))

