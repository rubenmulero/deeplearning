#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a trainable linear regression model in tensorflow to learn how it works


@author: Rubén Mulero
"""

import tensorflow as tf

# Defining model parameters
W = tf.Variable([.3], dtype=tf.float32)                 # Trainable parameters, same imput but different output
b = tf.Variable([-.3], dtype=tf.float32)
# Defining input and output values
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)                          # Promise to provide a value later

# loss function
loss = tf.reduce_sum(tf.square(linear_model - y))       # sum of squares
# Use gradient descent to optimize the model
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training the data
x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()                # initialize the saved dcelared values
sess = tf.Session()
sess.run(init)                                          # Reset values to wrong ones
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# evaqluate the training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
