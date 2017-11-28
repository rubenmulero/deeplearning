# -*- coding: utf-8 -*-
"""
In this file a Mnist based model is going to be used using advanced algorithms from tensorflow.


"""


__author__ = 'Rubén Mulero'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Loading mnisg data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Starting tensorflow as interactive session
sess = tf.InteractiveSession()

# Vamos a crear dos nodos que van a ser usados por el algoritmo de softmax
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Recuerda que éstos son valores que luego daremos nosotros, candiadtos a...

# Definimos los pesos que vamos a usar. En éste caso, son los tensoes que representan la cantidad de piexeles que tiene
# una imagen de mnis t la cantidad de valores que puede tomar ésta
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Recordemos que las variables son valores asignados a los nodos que pueden ser variados en la computación si es necesario

# estos tensores están inicializados como valores de 0. En W hemos creado una matriz de 784x10 porque tenemos
# 784 imputs que representan el número de pixeles de la imágen y tiene como salida 10 elementos que representa el valor
# que toma esa imágen (un número de 0 a 9)

# b es un vector de 10 elementos poruqe sólo tenemos 10 clases a clasificar (0 a 9)

# Para poder hacer uso de las varibls que hemos definido (nuestros nodos vaya), primero deben ser inicializadas utilizando
# una sesión. Éste paso coge un valor inicial que en éste caso serían matrices de 0.

sess.run(tf.global_variables_initializer())

# Implementamos el modelo de regresión
y = tf.matmul(x,W) + b

# Especificamos la loss fucntion
# La loss function nos indica cuanto de malo ha sido la predicción del modelo en un único ejemplo.
# La idea es intentar minimizar el valor obtenido en ésta función.
#
# Aquí vamos a hacer uso de un cross-entropy entre el target y la función softmax utilizada

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))



########### TRAINING
#
#   loading the needed optimizations to train section in TF
#
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # Esto es la inclusión de una operación más para optimizar el grafo

# Ahora vamos a lanzar el training paso a paso, haciendo un gradientdescent como hemos indicado arriba para optimizar resultados
#
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})



########### EVALUATE
#
#   Evaluation of the training set
#
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # Lista de booleanos con valores predecidos bien o mal
#
# Para conseguir que ésto sea legible, vamos a convertir esto en numeros flotantes basados en 0 o 1 y y hacer una media
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Finalmente imprimimos las soluciones
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# --> 0.9189

#### Como podemos ver no es precisamente un resultado aceptable. Vamos a usar ahora la potencia de las redes
# neuronales para poder trabajar como se merece con TF
########


# Vamos a generar una red neuronal convoluacionada pequeña para que trabaje y nos mejore los resultados
print("\nCreating a small convolutional neural network...............")

# 1º Inicialización de los pesos. Vamos a inicializar los pesos para poder construir la red. La ides es crear
# varios pesos con algo de ruido para evitar la generación de gradiemtes con un valor a 0

# Como se van a usar un montón de neuronas, la idea es crear dos procesos para poder inicializar todos los pesos
# de manera correcta. Generamos dos funciones, una para W y otra para la bias.


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Definiremos a continuaciñon las funciones de convolución y de pooling

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Implementamos la primera capa convolucional
# La red va a computar 32 características por cada parche de 5x5. El tensor va a tener un shape the 5,5,1,32
# donde las dos primeras dimensiones es el patc size (5x5) la tercera es el número de canales de input y la ltuima es el
# número de outputs

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Para aplicar el layer primero debemos the reshapear x a un tensor de 4d con la segunda y tercera dimensión
# que corresponden a la altura y anchura de la iamgen
x_image = tf.reshape(x, [-1, 28, 28, 1])


# Vamos a convolucionar la x image con el tensor de peso y el bias y a aplicar la función ReLU que contiene las redes neuronales
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)                 # reduce la imagen a 14x14


# Implementamos la segunda capa convolucional

W_conv2 = weight_variable([5, 5, 32, 64]) # Salida de 64 y entrada para las 32 entradas de la anterior salida
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Hemos conseguido reducir la imágen en 7x7 para la última capa de neuronas, vamos a aplicar ahora una red neuronal
# con 1024 neuroonas conectadas para podee procesar toda la imágen.
#
# Reshapeamos el tensor del pooling_2, y lo convertimos en un batch de vectores que será multiplicado por la matriz de pesos
# y el bias.. Después de aplica ReLU
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



# Para evitar que la red neuronal llegue a un estado de overfit, aplicaremos un dropout. Para hacer ésto, creamos
# un placehodlder

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout layer
# Adding the final layer as a readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

##### Training processs
#
#

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:                      # TF session separa el proceso de creación del grafo
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

