import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


#shape 주의
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

#matrix multiplication주의 2 = 들어오는값 1 = 나가는 값 / bias = 나가는 값
W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#바뀐 H(x), Cost function
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# if hypothesis>0.5 then true else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

#run Session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	feed = {X: x_data, Y:y_data}

	for step in range(10001):
		sess.run(train, feed_dict=feed)
		if step % 200 == 0:
			print(step, sess.run(cost, feed_dict=feed))

	h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
	print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


