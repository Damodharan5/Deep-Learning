"""

A simple derivative finder using tensorflow and validate the same thing using TensorBoard

"""
import tensorflow as tf
import numpy as np

X = np.arange(0,5,0.1)
cosx  = np.cos(X)
dcosx = -np.sin(X)

x = tf.Variable(initial_value=3.0)
y = tf.cos(x)


train = tf.train.GradientDescentOptimizer(0.01).minimize(y)
with tf.Session() as sess:
	writer = tf.summary.FileWriter('logs',sess.graph)
	writer.close()

