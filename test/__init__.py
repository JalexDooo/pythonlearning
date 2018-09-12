import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def conv_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积
filter1 = conv_variable([5, 5, 1, 32])
bias1 = bias_variable([32])
conv1 = conv2d(x_image, filter1)
h_conv1 = tf.nn.relu(conv1 + bias1)

# 池化
max_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
filter2 = conv_variable([5, 5, 32, 64])
bias2 = bias_variable([64])
conv2 = conv2d(max_pool1, filter2)
h_conv2 = tf.nn.relu(conv2 + bias2)

# 池化
max_pool2 = max_pool_2x2(h_conv2)

'''
# 第三层卷积
filter3 = conv_variable([5, 5, 64, 128])
bias3 = conv_variable([128])
conv3 = conv2d(max_pool2, filter3)
h_conv3 = tf.nn.sigmoid(conv3 + bias3)

# 池化
max_pool3 = max_pool_2x2(h_conv3)
'''

# 将卷积结果展开
h_pool2_flat = tf.reshape(max_pool2, [-1, 7 * 7 * 64])

# 隐含层1
w_fc1 = conv_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1))

# 隐含层2
w_fc2 = conv_variable([1024, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, w_fc2), b_fc2))

# 输出层
w_fc3 = conv_variable([128, 10])
b_fc3 = bias_variable([10])

# 输出层计算
y_res = tf.nn.softmax(tf.add(tf.matmul(h_fc2, w_fc3), b_fc3))

loss = -tf.reduce_sum(y_ * tf.log(y_res))
train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)

# 计算正确率
correct_prediction = tf.equal(tf.argmax(y_res, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

c = []

start_time = time.time()

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(200)

	if i % 2 == 0 :
		train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
		print("step %d, training accuracy %g" % (i, train_accuracy))
		c.append(train_accuracy)
		end_time = time.time()
		print("time: ", (end_time - start_time))
		start_time = end_time

	train_op.run(feed_dict={x: batch_xs, y_: batch_ys})

sess.close()
plt.plot(c)
plt.tight_layout()
plt.savefig('C:/Users/Jonty/Desktop/Project/trash/exam_image/13-14.png', dpi=200)

