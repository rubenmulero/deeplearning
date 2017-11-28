#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class, we are going to play with TF estimator, a high level tensorflow library to simplifies the machine
learning techniquies.


@author: Rubén Mulero
"""

# Import matematical libarries and tf

import numpy as np
import tensorflow as tf

# In this part, we are going to declare a list of features. In this case, we only have a unique numeric feature
# but it can be more dificult.

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Now, we are going to create an estimator. There are multiple estimators predefined in TF, in this case we are
# going to use a linear regression. The following line provides an estimator that provies a linear regression
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# Tensor flow provides different type of helpers methods to read and set up data sets.
# In this example, we usee two different datasets, one for training and other for evalating

# It is important to tell to the function how many batches of data (lotes de datos) (num_epoch) we want
# and how big each barch should be.

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 100 training steps by invoking the method and passing the training data set
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metris: %r", train_metrics)
print("eval metrics: %r", eval_metrics)

# The results returns a higher loss but they are close to 0

