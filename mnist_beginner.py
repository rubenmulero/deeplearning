#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a mnist python program for begginers. The idea is to learn how it works and how tensorflow can help
finding information in a image


@author: Rubén Mulero
"""

import tensorflow as tf

# Import the datasset information
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# En éste algoritmo tenemos dos tipos de elementos
#
# x --> La imágen procesada
# y --> El label que nos dice qué tipo de número aparece en la imágen

# Cada imágen tiene 28*28 pixeles

# Por lot tendremos dos tipos de tensores, el primero está en mnist.train.images, que contiene  las imágenes
# y los valores de los píxeles que lo conforman
#
# Aquí utilizamos un tensor con un shape de [55000, 784]]
# 55.000 representa el índice de las imágenes y 784 representa el índice de los pixeles que disponen (28*28)
#
# Por lo tanto, cada entrada contendrá un valor entre 0 y 1 que nos indicará la intensidad del pixel.

# El es el mnist.train.labels que contiene un tensor con un shape de [55000,10] donde 55000 representa el indice
# de las imágenes y 10 representa un valor de 0 a 9 para representan los números de 1 a 10.

# Cada entrada será un 0 o un 1 para indicar el label del número. Por ejemplo, para representar el valor del 3, el label
# sería --> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


# Para usar lo explicado, se va a hacer uso de softmax, un algoritmo que viene descrito en el ejemplo que estamos
# siguiendo

# Definimos un placeholder (valor al que prometemos que le daremos un valor fijo mas adelante). Con porofundidad de 784
x = tf.placeholder(tf.float32, [None, 784])

# Necesitamos establecer los pesos para el algoritmo softmax, recordemos que hemos definido el peso (W) y el bias (b)
# Lo vamos a establecer haciendo uso de las Variables para poder cambiar su valor cuando nos sea de nuestro interes

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Estas inicializaciones son full the 0 porque hemos usado tf.zeros

# Implementamos el modelo haciendo uso de softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Matmul multiplica las matrices x y W y luego sumamos el peso b.

######################## Training
#
# Para determinar el loss de éste modelo, vamos a usar el cross-entropy

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Aplicamos ahora el algoritmo de optimizacion para los tensores
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Lanzamos el modelo
sess = tf.InteractiveSession()
# Inizializamos las variables
tf.global_variables_initializer().run()

# Entrenamos el modelo
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)                  # Obtenemos un bach the 100 elementos aleatorios
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})       # runeamos el train pasandole los elementos del bach para las Variables

####################### Test
#
# argmax nos da el index de la entrada máxima de un tensor
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Esto nos va a dar varios true o false dependiendo de las predicciones correctas o no.
# Con esos valores vamos a intentar medir un accuracy
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
# Imprimimops los resultados
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

