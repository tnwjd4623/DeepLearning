import tensorflow as tf
import matplotlib.pyplot as plt

#Data Set
X = [1, 2, 3]
Y = [1, 2, 3]

# W & H(x)
W = tf.placeholder(tf.float32)
hypothesis = X*W

#cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y ) )

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

#그래프 구동
for i in range(-30, 50):		#-3 ~ 5
	feed_W = i * 0.1
	curr_cost, curr_W = sess.run([cost, W], feed_dict = {W: feed_W} )
	W_val.append(curr_W)
	cost_val.append(curr_cost)


# Show Graph
plt.plot(W_val, cost_val) # X = W_val / Y = cost_val
plt.show()
