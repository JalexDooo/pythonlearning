from random import shuffle
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import glob
import sys


shuffle_data = True
image_path = "C:\\Users\\Jonty\\Desktop\\Project\\TMExample\\*.png"
train_filename = "src/train.tfrecords"


def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def write_tfrecords():
    addrs = glob.glob(image_path)
    labels = np.zeros(len(addrs))
    print(labels)
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    train_addrs = addrs[0:int(0.7*len(addrs))]
    train_labels = labels[0:int(0.7*len(labels))]

    val_addrs = addrs[int(0.7*len(addrs)) : int(0.9*len(addrs))]
    val_labels = labels[int(0.7*len(labels)) : int(0.9*len(labels))]

    test_addrs = addrs[int(0.9*len(addrs)):]
    test_labels = labels[int(0.9*len(labels)):]

    writer = tf.python_io.TFRecordWriter(train_filename)

    for i in range(len(train_addrs)):

        if not i % 1000:
            print('Train data : {}/{}'.format(i, len(train_addrs)))
            sys.stdout.flush()

        img = load_image(train_addrs[i])
        label = train_labels[i]

        feature = {
            'train/label': _float_feature(label),
            'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


if __name__ == "__main__":
    write_tfrecords()
