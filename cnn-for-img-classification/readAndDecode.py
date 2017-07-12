# -*- coding: utf-8 -*-
import tensorflow as tf


def _read_and_decode(filename, width=120, height=213, batch_size=64):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, shape=[width, height, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.string)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                num_threads=4,
                                                batch_size=batch_size,
                                                capacity=50000,
                                                min_after_dequeue=10000)

    return img_batch, label_batch
