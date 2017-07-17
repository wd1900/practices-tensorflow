# -*- coding: utf-8 -*-
import tensorflow as tf


def read_my_file_format(filename_queue, height, width):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, shape=[height, width, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.string)
    return img, label


def _read_and_decode(filename, height=213, width=120, batch_size=64):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    print(filename)
    img, label = read_my_file_format(filename_queue, height, width)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                num_threads=4,
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue)

    return img_batch, label_batch


def distorted_img(img_batch, height, width):
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(img_batch, [height, width, 3])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])

    return float_image

