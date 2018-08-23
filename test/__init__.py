import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

threshold = 1.0e-2
x_data = np.random.randn(100).astype(np.float32)
y_data = x_data * 0.3 + 0.15

weight = tf.Variable(1.)
bias = tf.Variable(1.)
x_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y_model = tf.add(tf.multiply(x_, weight), bias)

loss = tf.reduce_mean(pow((y_model - y_), 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

flag = True
while flag:
    for (x, y) in zip(x_data, y_data):
        sess.run(train_op, feed_dict={x_: x, y_: y})
        print('weight={}|bias={}'.format(weight.eval(sess), bias.eval(sess)))
    if sess.run(loss, feed_dict={x_: x_data, y_: y_data}) <= threshold:
        flag = False
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, sess.run(weight) * x_data + sess.run(bias), label='Fitted line')
plt.legend()
plt.show()
