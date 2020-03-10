import tensorflow as tf

# data non-changable var
a = tf.constant(1)
b = tf.constant(2)
matrix_a = tf.constant([[1,2],[3,4]])
matrix_b = tf.constant([[2,0],[0,2]])

# function
c = tf.add(a, b)
# Eager-execution without session
tf.print(c)
tf.print(tf.matmul(matrix_a, matrix_b))
# Ordinary execution without Tensorflow library
print(c.numpy())