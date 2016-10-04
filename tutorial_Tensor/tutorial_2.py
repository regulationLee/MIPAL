# CNN with 5 convolution layer, max pooling, dropout

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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



# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
sess = tf.InteractiveSession()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

learning_rate = 0.0001
training_epochs = 25
batch_size = 100
dropout_r = 0.7
display_step = 1

# tensorflow graph input
X = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


# Set model weights and biases
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024]   )

# Reshape image for apply CNN
x_image = tf.reshape(X, [-1, 28, 28, 1])

dropout_rate = tf.placeholder(tf.float32)

# Construct model
# 1st layer w/maxpooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd layerw/maxpooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely(fully connect) connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
h_fc1_drop = tf.nn.dropout(h_fc1, dropout_rate)

# final output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# Define activation function(?)  // matmul : matrix multiply
activation = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define cost function and optimizer
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(activation), reduction_indices=[1])) # cross-entropy
        # reduce_mean : Computes the mean of elements across dimensions of a tensor
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Gradient descent

# Before starting, initialize the variables.
sess.run(tf.initialize_all_variables())

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)

    # Fit the line
    for step in xrange(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        optimizer.run(feed_dict={X: batch_xs, y_: batch_ys, dropout_rate: dropout_r})

        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, y_: batch_ys, dropout_rate: dropout_r})/total_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch + 1), "cost=", '%.9F' %(avg_cost))

print ("Optimization Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y_, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels, dropout_rate: 1}))

