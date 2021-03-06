# -*- coding: utf-8 -*-
import tensorflow as tf

from readAndDecodeBatch import _read_and_decode
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tfrecord_dir", required=True, help="where to get input tfrecords")
a = parser.parse_args()

img, label = _read_and_decode(a.tfrecord_dir, width=120, height=213)

# 使用shuffle_batch可以随机打乱输入
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 num_threads=4,
#                                                 batch_size=64, 
#                                                 capacity=50000,
#                                                 min_after_dequeue=10000)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        val, l = sess.run([img, label])
        print(val.shape)
        print(l)
    coord.request_stop()
    coord.join(threads)
