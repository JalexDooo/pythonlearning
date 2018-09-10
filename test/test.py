import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Jonty\\Pictures\\Saved Pictures\\lena1.jpg")
img = np.array(img, dtype=np.float32)
x_image = tf.reshape(img, [1, 300, 238, 3])

filter = tf.Variable(tf.ones([3, 3, 3, 1]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

res = tf.nn.conv2d(x_image, filter, strides=[1, 1, 1, 1], padding='SAME')
res = tf.nn.max_pool(res, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
res_image = sess.run(tf.reshape(res, [150, 119]))+1

tmp = tf.nn.conv2d(x_image, filter, strides=[1, 1, 1, 1], padding='SAME')
ori_image = sess.run(tf.reshape(tmp, [300, 238]))%256+1

cv2.imshow('testimage', ori_image.astype('uint8'))
cv2.waitKey()
cv2.imshow("lena", res_image.astype('uint8'))
cv2.waitKey()

