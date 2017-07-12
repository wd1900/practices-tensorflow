from __future__ import print_function
from readAndDecode import _read_and_decode

import tensorflow as tf
import numpy as np
import argparse
import math
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--tfrecord", required=True, help="where to get train tfrecords")
parser.add_argument("--test_tfrecord", required=True, help="where to get test tfrecords")
parser.add_argument("--origin_dir", required=True, help="orginal images directory")
# Parameters
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
parser.add_argument("--training_iters", type=int, default=2000, help="training_iters")
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--display_step", type=int, default=10,  help="display_step")
parser.add_argument("--img_width", type=int, required=True,  help="the width of image")
parser.add_argument("--img_height", type=int, required=True, help="the height of image")
# Network Parameters
parser.add_argument("--n_classes", type=int, default=2,  help="n_classes")
parser.add_argument("--dropout", type=float, default=0.75,  help="n_classes")

a = parser.parse_args()

fc_layer_w = math.ceil(math.ceil(a.img_width/2)/2)
fc_layer_h = math.ceil(math.ceil(a.img_height/2)/2)

# Find every directory name in the orginal images  directory (dance,nodance ...)
labels = list(map(lambda c: c.split("/")[-1], glob.glob(a.origin_dir + "/*")))
labels_onehot = np.eye(a.n_classes, dtype=int)
labels_onehot_dict = {}
for i, el in enumerate(labels):
    labels_onehot_dict[el] = labels_onehot[i]
print(labels_onehot_dict)

# tf Graph input
x = tf.placeholder(tf.float32, [None, a.img_width, a.img_height, 1])
y = tf.placeholder(tf.float32, [None, a.n_classes])
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W, b, strides=1):
    # Conv2D warpper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k,1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([fc_layer_w*fc_layer_h*64, 1024])),
    # 1024 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, a.n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([a.n_classes]))
}

pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=a.learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

img_batch, label_batch = _read_and_decode(a.tfrecord, a.img_width, a.img_height, a.batch_size)
img_batch_test, label_batch_test = _read_and_decode(a.test_tfrecord, a.img_width, a.img_height, a.batch_size)


# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    batch_x, batch_y = sess.run([img_batch, label_batch])
    batch_y = np.array([labels_onehot_dict[x.decode()] for x in batch_y], dtype=int)

    batch_x_test, batch_y_test = sess.run([img_batch_test, label_batch_test])
    batch_y_test = np.array([labels_onehot_dict[x.decode()] for x in batch_y_test], dtype=int)
    print(batch_y.shape)
    for i in range(a.training_iters):
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: a.dropout})
        if i % a.display_step == 0:
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1})
            print("test Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    coord.request_stop()
    coord.join(threads)

    print("Optimization Finished!")

