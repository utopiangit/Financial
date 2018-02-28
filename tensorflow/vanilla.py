# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def bs_mc(underlying, strike, volatility, tau, n_path = 1000, n_div = 100):
#    path = tf.random_normal([n_path, n_div]) * volatility * tf.sqrt(tau / n_div)
    path = np.random.randn(n_path, n_div)
    path = path * volatility * tf.sqrt(tau / n_div)
    drift = (-volatility**2 / 2.) * tau
    ST = tf.exp(tf.reduce_sum(path, axis = 1) + drift) * underlying
    payoff = tf.maximum(ST - strike, 0)
    return tf.reduce_mean(payoff)

def black_call(fwd, strike, volatility, tau):
    '''
    Calculate Call Option Price Under Black Model
    '''
    dist = tf.distributions.Normal(loc=0., scale=1.)
    return (fwd * dist.cdf(_d1(fwd, strike, volatility, tau))
            - strike * dist.cdf(_d2(fwd, strike, volatility, tau)))

def _d1(fwd, strike, volatility, tau):
    return (tf.log(fwd/strike) +  volatility**2 / 2 * tau)/(volatility * tf.sqrt(tau))

def _d2(fwd, strike, volatility, tau):
    return (tf.log(fwd / strike)  - volatility**2 / 2 * tau) / (volatility * tf.sqrt(tau))

def black_delta(fwd, strike, volatility, tau):
    '''
    Calculate Call Option Delta Under Black Model
    '''
    dist = tf.distributions.Normal(loc=0., scale=1.)
    return dist.cdf(_d1(fwd, strike, volatility, tau))


if __name__ == '__main__':
    S = tf.Variable(100.)
#    K = tf.constant(100.)
    K = 100
    tau = tf.Variable(1.)
    volatility = tf.Variable(0.1)

#    v = bs_mc(S, K, volatility, tau)
#    delta = tf.gradients(v, S)[0]
#    vega = tf.gradients(v, volatility)[0]
#    # モンテカルロで計算すると、デルタに対する原資産の寄与がなくなる。
#    # 次の行をrunするとエラー
#    # gamma = tf.gradients(delta, S)[0]
#    # vannaも計算はできるが、結果がずれる
#    vanna = tf.gradients(vega, S)[0]

    # 解析式なら二回微分まで問題なく計算できる
    v_a = black_call(S, K, volatility, tau)
    delta_a = tf.gradients(v_a, S)[0]
    vega_a = tf.gradients(v_a, volatility)[0]
    gamma_a = tf.gradients(delta_a, S)[0]
    vanna_a = tf.gradients(vega_a, S)[0]
    theta_a = tf.gradients(v_a, tau)[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#        print('v :', sess.run(v))
#        print('delta :', sess.run(delta))
#        print('vega :', sess.run(vega))
#        print('vanna :', sess.run(vanna))

        print('v_a :', sess.run(v_a))
        print('delta_a :', sess.run(delta_a))
        print('vega_a :', sess.run(vega_a))
        print('gamma_a :', sess.run(gamma_a))
        print('vanna_a :', sess.run(vanna_a))
        print('theta_a :', sess.run(theta_a))
