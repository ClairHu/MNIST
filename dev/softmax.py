import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data	# a built-in utility for retrieving the dataset on the fly



DATA_DIR = '/tmp/data'	# the location we wish the data to be saved to locally
NUM_STEPS = 1000
MINIBATCH_SIZE = 100


data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])	# size [None, 784] means that each image is of size 784 (28×28 pixels unrolled into a single vector), and None is an indicator that we are not currently specifying how many of these images we will use at once:
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10]) # the true and predicted labels
y_pred = tf.matmul(x, W)

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))


gd_step = tf.train.GradientDescentOptimizer(0.56).minimize(cross_entropy) # how we are going to train it (i.e., how we are going to minimize the loss function)
# 0.5 is the learning rate, controlling how fast our gradient descent optimizer shifts model weights to reduce overall loss

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

    # Train
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images, 
                                        y_true: data.test.labels})

print ("Accuracy: {:.4}%".format(ans*100))