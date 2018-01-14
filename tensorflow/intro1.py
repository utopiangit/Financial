# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    x = np.random.rand(100).astype(np.float) 
    y = x * 0.1 + 0.3

    W = tf.Variable(tf.random_uniform([1], -1., 1.))
    b = tf.Variable(tf.zeros([1]))
    y_pred = W * x + b

    loss = tf.reduce_mean(tf.square(y_pred - y))
    optimizer = tf.train.GradientDescentOptimizer(0.3)
    train = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
    
    for i in range(200):
        sess.run(train)
        if i % 20 == 0:
            print(sess.run(W), sess.run(b))
    
