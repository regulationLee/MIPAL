# CNN with 5 convolution layer, max pooling, dropout

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# Create the model
X = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define functions for multilayer convolutional network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    # conv2d : compute a 2-D convolution given 4-D input and filter tensor


def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



# First convolution layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(X, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolution layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely(fully connect) connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
dropout_rate = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, dropout_rate)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# Define activation function(?)  // matmul : matrix multiply
activation = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Cost function : Cross entropy  // reduce_mean : Computes the mean of elements across dimensions of a tensor.
cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(activation), reduction_indices=[1]))

# Gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y_, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Before starting, initialize the variables.
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # Fit the line
        for step in xrange(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            sess.run(optimizer, feed_dict={X: batch_xs, y_: batch_ys, dropout_rate: 0.7})

            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, y_: batch_ys, dropout_rate: 0.7})/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9F}".format(avg_cost), "training accuracy:",
                  accuracy.eval({X: batch_xs, y_: batch_ys, dropout_rate: 0.7}))

        print ("Optimization Finished!")

    print("Accuracy: ", accuracy.eval({X: mnist.test.images, y_: mnist.test.labels, dropout_rate: 1}))

