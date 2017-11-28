#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:40:52 2017

@author: elektro
"""

import tensorflow as tf


# Building a computacional graph, a series of TS operations
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # Implicit a float32 dtype
# print(node1, node2)

# The above are nodes that will ouptut that values when evaluating
# we only define how they work
sess = tf.Session()
print(sess.run([node1, node2]))


# Building more complicated tensors, by reusing the previous used tensors
node3 = tf.add(node1, node2)
print('node3: ', node3)
print('sess.run(node3): ', sess.run(node3))

# Using placeholder to promise later values to the node
# We are defining two tensor of float32 wiouth using a values
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # --> using + is a shortcut of using tf.add(a,b)

# The idea is to later give to this two tensor a values to obtain the 
# adder_value operation result
print(sess.run(adder_node, {a:3,b:4.5}))
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

# This placeholder operation can be done more complicated if we want 
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a:3, b:4.5}))

# Another way to create nodes is using variables that can be used later
# in machine learning we want to change the imput values to test different
# arbitrary values

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# Tf variable allow to not initiaize the imput values.If a variable is initialized
# you cannon change its value. 
# The command to initialize the saved dcelared values is:
init = tf.global_variables_initializer()
# At this point the varibles are initialized
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

# OK, so now we know hot to create different models but we need to know exactly
# how to evaluate them. The best way is using a loss function
# based in a regression model
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Re-assing the values of W and b to different ones
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

#######
# Optimizers allows to slowly change each variable to minimize the loss function
#
# to use this we have the tf.gradients to do it.
#
# Example
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)          # Calls to the initializer and resets the values to incorrect defaults
for i in range(1000):
    sess.run(train, {x:[1,2,3,4,], y: [0,-1,-2,-3]})
    
print(sess.run([W, b]))


