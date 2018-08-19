import tensorflow as tf
import numpy as np
from PIL import Image
import os

path = "C:\\Users\\Jonty\\Desktop\\Project\\天猫工商信息执照"


def test1():

    filename = os.listdir(path)
    strText = ""

    with open("C:\\Users\\Jonty\\Desktop\\Project\\trash\\test.csv", "w") as fid:
        for a in range(len(filename)):
            strText = path+os.sep+filename[a] + "," + filename[a] + "\n"
            fid.write(strText)
        fid.close()


def test2():
    image_add_list = []
    image_label_list = []
    with open("C:\\Users\\Jonty\\Desktop\\Project\\trash\\test.csv") as fid:
        for image in fid.readlines():
            image_add_list.append(image.strip().split(",")[0])
            image_label_list.append(image.strip().split(",")[1])

        print(image_add_list)
        print(image_label_list)

    img = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file('C:\\Users\\Jonty\\Desktop\\Project\\天猫工商信息执照\\1.jpg'), channels=1)
                                       , dtype = tf.float32)
    print(img)


def test3():
    a_data = 0.834
    b_data = [17]
    c_data = np.array([[0, 1, 2], [3, 4, 5]])
    c = c_data.astype(np.uint8)
    c_raw = c.tostring()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'a': tf.train.Feature(
                    float_list=tf.train.FloatList(value=[a_data])
                ),
                'b': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=b_data)
                ),
                'c': tf.train.Feature(
                    btypes_list=tf.train.BytesList(value=[c_raw])
                )
            }

        )
    )
    return


def test4():
    writer = tf.python_io.TFRecordWriter("trainArray.tfrecords")
    for _ in range(100):
        randomArray = np.random.random((1, 3))
        print(randomArray)
        array_raw = randomArray.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    return


def test5():
    filename_queue = tf.train.string_input_producer(["trainArray.tfrecords"], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int32),
            'img_raw': tf.FixedLenFeature([], tf.float32)
        }
    )
    label = features["label"]
    img_raw = features['img_raw']
    print(img_raw)
    return


def test6():
    filenames = os.listdir(path)
    writer = tf.python_io.TFRecordWriter("src/train.tfrecords")
    sep = ['jpg', 'png', 'jpeg']
    for name in filenames:
        img_path = path + os.sep + name
        img_name, img_sep = name.split('.')
        if img_sep not in sep:
            continue
        img = Image.open(img_path)
        img = img.resize((500, 500))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }
        ))
        writer.write(example.SerializeToString())
    return


if __name__ == "__main__":
    test6()
