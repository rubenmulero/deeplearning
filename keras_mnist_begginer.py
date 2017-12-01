#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

https://elitedatascience.com/keras-tutorial-deep-learning-in-python

This is a basic tutorial of how to use Keras to create a convolutional neural network.


# Here are are using the Theano backend. In general we can configure this by modifying the ~.keras/keras.json

@author: Rubén Mulero
"""


import numpy as np
# Creating a random seed for reprodicibility
np.random.seed(123)
# Import Keras and it's layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# Import the Convolutional Neuronal Networks
from keras.layers import Convolution2D, MaxPooling2D
# Import some additional utils
from keras.utils import np_utils
# Loading Mnist dataset from KERAS
from keras.datasets import mnist
# We set up the image dimension ordering to use it like theano
from keras import backend as K
K.set_image_dim_ordering('th')


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Look the shape of the dataset
print(X_train.shape)        # 60000 samples of 28x28

# We are going to plot this examples using matplotlib
from matplotlib import pyplot as plt

plt.imshow(X_train[0])
plt.colorbar()
plt.title('plot of the first training set')
# plt.show()

# We need to declare the depth of the image in Theano library. In general an image contains a RGB channel
# that is a depth of = 3. but in other scenarios we have more!. In thi case mnist only contains 1 channel so
# we need to define a depth = 1

# We are going to transform our dataset to ahve the following shape --> (n, depth, width, height)

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# We are going to print again its dimensions to confirm
print(X_train.shape)

# OK, so now we are going to finish to preprocess this data giving it a float32 value and making a normalization
# to put in in range of [0,1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

########## label data
# Lets have a look on or label data
print(y_train.shape)
# We see that this can be problematic because we need to have 10 different clases and this shape doesn't represent
# that condition. We are going to have a look inside the the first 10 training samples:
print(y_train[:10])
# So the problem is that we need to convert the 1-dimensional class to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# now we can see that our shape its correct. We have 60000 samples with 10 possible clasifications (from 0 to 9)
print(Y_train.shape)

########### Model architecture
# After preparing the data, we are going to configure our CNN architecture.
#
model = Sequential()
# Defining the input layer
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28)))

# The input shape is the information of each image -> 1 depth (black o white), 28 width and 28 height
# We need to put the input shape values ordered to feed correctly the convolutional neuronal network
# The third parameter represent -> the number of convolution filters, the number of rows in each convolution the number
# of columns in each convolution.
# We are going to confirm this model printing its shape
print(model.output_shape)

# Now, we are going to add more layer to our CNN
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))                    # kill 25% of the inputs to avoid overfilling

# Now, after adding the CNN, we are going to add the Dense layers to complete or fully connected network.
model.add(Flatten())                                # Convert to n-dimensional layers to 1-dimensional layer to be used by the Dense layer.
                                                    # We need to have only vector with all involved weights to be used by the dense layers
model.add(Dense(128, activation='relu'))            # The first parameter is the output size
model.add(Dropout(0.5))                             # Again, we will the 50% of the neurons
model.add(Dense(10, activation='softmax'))          # Output size = 10 the total number of digits to classify


########### Model compilation
# After having defined how it looks like our model. We are going to compile it and add aditional fields like:
#
#   * training optimizer
#   * loss function
#   * metrics
#   * ........
model.compile(loss='categorical_crossentropy',      # Using the categorial cross entropy for functions that are categorial (We defined earlier)
              optimizer='adam',                     # AdamOptimizer
              metrics=['accuracy'])                 # User accuracy as metric to evaluate the quality of the model

########### Fitting the model on training data
model.fit(X_train, Y_train,
          batch_size=32, epochs=10, verbose=1)
# Here, we defined the batch size 32 inputs
#
# And the number of epoch to use in the training set (10 times)
# The verbose function allows to know the information of the training

########### Evaluating the model
# Once we have trained our model. We can evaluate it to see the accuracy
score = model.evaluate(X_test, Y_test, verbose=0)
