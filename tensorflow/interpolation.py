# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def monotone_convex(t, grids, discount_factors):
    '''Interpolate discount factor with Monotone Convex by Hagan and West(2006)

    Arguments:
       t : array_like
           Points to evaluate the interpolant discount factor.
       grids : array_like
           Time grids that fixed discount factors are given.
       discount_factors : array_like
           Discount factors given fixed.

    Return:
        y : array_like
        Interpolated discount factors.
    '''
#    if not(isinstance(t, np.ndarray)):
#        t = tf.array([t])
#    if not(tf.equal(grids[0], 0)):
#        grids = tf.append([0], grids)
#        discount_factors = np.append(1, discount_factors)

    shifted = (grids[:-1] + grids[1:]) / 2.
    indice = tf.argmin(tf.abs(
        tf.reshape(t, [-1, 1]) - tf.reshape(shifted, [1, -1])),
        axis = 1) + 1

    f_discrete = -(tf.log(discount_factors[1:] / discount_factors[:-1])
                   / (grids[1:] - grids[:-1]))
    f = ((grids[1:-1] -  grids[:-2]) * f_discrete[1:]
        + (grids[2:] -  grids[1:-1]) * f_discrete[:-1]) / (grids[2:] - grids[:-2])
    f0 = (3. * f_discrete[0] - f[0]) / 2.
    fn = (3. * f_discrete[-1] - f[-1]) / 2.
    f = tf.concat([[f0], f, [fn]], axis = 0)

#    print('grids :', grids)
#    print('f_discrete :', f_discrete)
#    print('f :', f)

    fi = tf.gather(tf.reshape(f, [1, -1]),
                   tf.reshape(indice, [-1, 1]),
                   axis = 1)
    fi_prev = tf.gather(tf.reshape(f, [1, -1]),
                        tf.reshape(indice - 1, [-1, 1]),
                        axis = 1)
    fdi = tf.gather(tf.reshape(f_discrete, [1, -1]),
                    tf.reshape(indice - 1, [-1, 1]),
                    axis = 1)
    g0 = fi_prev - fdi
    g1 = fi - fdi
    ti = tf.gather(tf.reshape(grids, [1, -1]),
                   tf.reshape(indice, [-1, 1]),
                   axis = 1)
    ti_prev = tf.gather(tf.reshape(grids, [1, -1]),
                        tf.reshape(indice - 1, [-1, 1]),
                        axis = 1)

    t = tf.reshape(t, [-1])
    ti = tf.reshape(ti, [-1])
    ti_prev = tf.reshape(ti_prev, [-1])
    g0 = tf.reshape(g0, [-1])
    g1 = tf.reshape(g1, [-1])

    x = (t - ti_prev) / (ti - ti_prev)
    def integrate_g(x, g0, g1):
        # region(i)
        Gi = g0 * (x - 2. * x**2  + x**3) + g1 * (-x**2 + x**3)
        # region(ii)
        eta = 1 + 3. * g0 / (g1 - g0)
        Gii = g0 * x + tf.where(x < eta, tf.zeros(eta.shape), (g1 - g0) * (x - eta)**3 / (1 - eta)**2 / 3.)
        # region(iii)
        eta = 3. * g1 / (g1 - g0)
        Giiia = g1 * x + (g0 - g1) * ((x - eta)**3 + eta**3) / (3 * eta**2)
        Giiib = g1 * (x - 1) 
        Giii = tf.where(x < eta, Giiia, Giiib)
        # region(iv)
        eta = g1 / (g0 + g1)
        A = -g0 * g1 / (g0 + g1)
        Giva = A * x + (g0 - A) * ((x - eta)**3 + eta**3) / (3 * eta**2)
        Givb = A * x + (g0 - A) * eta / 3 + (g1 - A) * (x - eta)**3 / (3 * (1 - eta)**2)
        Giv = tf.where(x < eta, Giva, Givb)
        G = [Gi, Gii, Giii, Giv]
        return G

    g_integrated = integrate_g(x, g0, g1)
#    print('g integrated :', g_integrated)
    G = tf.where(tf.logical_or(tf.equal(x, 0), tf.equal(x, 1)),
                 tf.zeros(x.shape),
                 tf.where((g0 + 2 * g1) * (2 * g0 + g1) < 0,
                          g_integrated[0],
                          tf.where(g0 * (2 * g0 + g1) <= 0,
                                   g_integrated[1],
                                   tf.where(g1 * (g0 + 2 * g1) <= 0,
                                            g_integrated[2],
                                            g_integrated[3]))))
    df = tf.gather(tf.reshape(discount_factors, [1, -1]),
                   tf.reshape(indice - 1, [-1, 1]),
                   axis = 1)
    df = tf.reshape(df, [-1])
    fdi = tf.reshape(fdi, [-1])
    return df * tf.exp(-G * (ti - ti_prev) - fdi * (t - ti_prev))


def linear(t, grids, discount_factors):
    '''Interpolate discount factor with Linear Interpolation
    Arguments:
       t : array_like
           Points to evaluate the interpolant discount factor.
       grids : array_like
           Time grids that fixed discount factors are given.
       discount_factors : array_like
           Discount factors given fixed.

    Return:
        y : array_like
        Interpolated discount factors.
    '''

    shifted = (grids[:-1] + grids[1:]) / 2.
    index = tf.argmin(tf.abs(
        tf.reshape(t, [-1, 1]) - tf.reshape(shifted, [1, -1])),
        axis = 1) + 1
#    tiled = tf.reshape(tf.tile(discount_factors, [tf.size(index)]),
#                      [tf.size(index), -1])
#    print('df :', tf.reshape(discount_factors,[-1, 1]))
#    print('index :', ind)
#    return index
#    return tf.gather(tf.reshape(discount_factors,[1, -1]),
#                     tf.reshape(index, [-1, 1]),
#                     axis = 1)
#    return tf.reduce_max(tiled * one_hot, axis = 1)

    df = tf.gather(tf.reshape(discount_factors,[1, -1]),
                     tf.reshape(index, [-1, 1]),
                     axis = 1)
    df_prev = tf.gather(tf.reshape(discount_factors,[1, -1]),
                        tf.reshape(index - 1, [-1, 1]),
                        axis = 1)
    ti = tf.gather(tf.reshape(grids,[1, -1]),
                   tf.reshape(index, [-1, 1]),
                   axis = 1)
    ti_prev = tf.gather(tf.reshape(grids,[1, -1]),
                        tf.reshape(index - 1, [-1, 1]),
                        axis = 1)
    grad = tf.reshape((df - df_prev) / (ti - ti_prev), [-1])
    df_prev = tf.reshape(df_prev, [-1])
    ti_prev = tf.reshape(ti_prev, [-1])
#    print('grad', grad.shape)
#    print('t', t.shape)
#    print('ti_prev', ti_prev.shape)
#    print('df_prev', df_prev.shape)
    return grad * (t - ti_prev) + df_prev

def log_linear(t, grids, discount_factors):
    '''Interpolate discount factor with Log Linear

    Arguments:
       t : array_like
           Points to evaluate the interpolant discount factor.
       grids : array_like
           Time grids that fixed discount factors are given.
       discount_factors : array_like
           Discount factors given fixed.

    Return:
        y : array_like
        Interpolated discount factors.
    '''
    return tf.exp(linear(t, grids, tf.log(discount_factors)))



def _test_curve_shape():
    import matplotlib.pyplot as plt

    grids = np.array([1,2,3,4, 5], dtype = np.float32)
    dfs = tf.Variable([0.993, 0.98,0.97,0.95, 0.94], dtype = tf.float32)

    ts = np.arange(0, 6, 1. / 365., dtype = np.float32)

    df_ll = log_linear(ts, grids, dfs)
    df_mc = monotone_convex(ts, grids, dfs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        DF_ll = sess.run(df_ll)
        DF_mc = sess.run(df_mc)
#        print(sess.run(df))
    print('DF_mc', DF_mc.shape)
    print('ts', ts)
    plt.plot(ts, DF_ll, label = 'log linear')
    plt.plot(ts, DF_mc, label = 'monotone convex')
    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100))
    plt.show()

    fwd_ll = -np.log(DF_ll[1:] / DF_ll[:-1]) / (ts[1:] - ts[:-1])
    fwd_mc = -np.log(DF_mc[1:] / DF_mc[:-1]) / (ts[1:] - ts[:-1])
    #print(fwd)
    plt.plot(ts[:-1], fwd_ll, label = 'log linear')
    plt.plot(ts[:-1], fwd_mc, label = 'monotone convex')
    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100))
    plt.show()

def _test_build_curve():
    import matplotlib.pyplot as plt
    import time

#    grids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype = np.float32)
#    dfs = tf.Variable([1, 0.993, 0.98,0.97,0.95, 0.94, 0.93, 0.92, 0.91,0.9, 0.89], dtype = tf.float32)
#    terms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                     dtype = np.float32)
#    rate = tf.constant([0.01, 0.012, 0.015, 0.015, 0.017, 0.018, 0.018, 0.019, 0.019, 0.02],
#                       dtype = np.float32)
    terms = tf.constant([1, 2, 3, 4, 5],
                     dtype = np.float32)
    rate = tf.constant([0.01, 0.012, 0.015, 0.015, 0.017],
                       dtype = np.float32)

    grids = tf.constant([0, 1, 2, 3, 4, 5], dtype = np.float32)
    dfs = tf.Variable([0.993, 0.98,0.97,0.95, 0.94], dtype = tf.float32)
    df0 = tf.constant([1], dtype = tf.float32)
    dfs = tf.concat([df0, dfs], axis = 0)

    t0 = time.time()
    # curve = log_linear
    curve = monotone_convex
    rate_calc = -tf.log(curve(terms, grids, dfs)) / terms
    loss = tf.reduce_mean(tf.square(rate - rate_calc))
    optimizer = tf.train.AdamOptimizer(0.5).minimize(loss)

    # A list of `sum(d rate_calc/d dfs)` for each df in `dfs`.
    jacobian = []
    for i in range(rate_calc.shape[0]):
        jacobian.append(tf.gradients(rate_calc[i], dfs))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            sess.run(optimizer)

        t1 = time.time()
        dfs_calib = sess.run(dfs)
        print('df@regular :', dfs_calib)
        print('rate calc :', sess.run(rate_calc))
        print('time :', t1 - t0)

        t2 = time.time()
        jacobian_value = sess.run(jacobian)
        t3 = time.time()
        print('jacobian :', jacobian_value)
        print('time :', t3 - t2)


    ts = np.arange(0, 10, 1. / 365., dtype = np.float32)

    df = curve(ts, grids, dfs_calib)
    t4 = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        df = sess.run(df)
#        print(sess.run(df))
    t5 = time.time()
    print('time :', t5 - t4)
#    plt.plot(ts, df, label = 'log linear')
#    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100))
#    plt.show()


    fwd = -np.log(df[1:] / df[:-1]) / (ts[1:] - ts[:-1])
    plt.plot(ts[:-1], fwd, label = 'monotone convex')
    plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100))
    plt.show()


def broadcastable_where(condition, x=None, y=None, *args, **kwargs):
    if x is None and y is None:
        return tf.where(condition, x, y, *args, **kwargs)
    else:
        _shape = tf.broadcast_dynamic_shape(tf.shape(condition), tf.shape(x))
        _broadcaster = tf.ones(_shape)
        return tf.where(
            condition & (_broadcaster > 0.0),
            x * _broadcaster,
            y * _broadcaster,
            *args, **kwargs
        )

if __name__ == '__main__':
#    _test_curve_shape()
    _test_build_curve()
