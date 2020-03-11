# import modules
import tensorflow as tf
import numpy as np

# Dataset to test AND operation
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[0],[0],[1]], dtype=np.float32)

"""
How the operation works
0 : 0 -> 0
0 : 1 -> 0
1 : 0 -> 0
1 : 1 -> 1
"""

# X placeholder will have 2 dimensional tensor as input
X = tf.placeholder(tf.float32, [None, 2], name="x-input")
# Y placeholder will have 1 dimensional tensor as input
Y = tf.placeholder(tf.float32, [None, 1], name="y-input")

# These two variables are going to change as optimization goes on
W = tf.Variable(tf.random_normal([2,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# This formula introduces how the sigmoid function consists
Hypothesis = tf.sigmoid(tf.matmul(X, W)+b)

# Now, since we all have essential vars and function, let's construct the loss funciton -> Cross-Entrophy function
cost = (-1)*tf.reduce_mean(Y*tf.log(Hypothesis)+(1-Y)*tf.log(1-Hypothesis))

# This is the problem that I have to redefine the whole mechanism
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Now, time to calculuate accuracy & prediction
prediction = tf.cast(Hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

# Construct session & design graph
with tf.Session() as sess:
    # initialize tf vars
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    # Reporting Accuracy
    h, c, a = sess.run([Hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
    print("Hypothesis : ", h, "Correct : ", c, "Accuracy : ", a)
