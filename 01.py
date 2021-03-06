import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()

#配置训练数据的地址
parser.add_argument('--data_dir', type=str, default='', help='input data path')
#配置模型保存地址
parser.add_argument('--model_dir', type=str, default='', help='output model path')

FLAGS, _ = parser.parse_known_args()
print(FLAGS)

mnist = input_data.read_data_sets("test/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.placeholder(tf.float32, [None, 10])

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean( -tf.reduce_sum( y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
