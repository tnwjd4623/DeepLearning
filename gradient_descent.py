import tensorflow as tf

#Data Set
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Weight, 넘겨줄 X , Y
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


#H(x)
hypothesis = X * W

#cost function
cost = tf.reduce_sum(tf.square(hypothesis - Y ) )

#Minimize
learning_rate = 0.1				#Constant
gradient = tf.reduce_mean(( W * X - Y ) * X )	#기울기
descent = W - learning_rate * gradient		#기울기에 따른 W 변화
update = W.assign(descent)			# W = descent --> W가 가장 밑으로 수렴함

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#W가 1에 가까워지면 맞음
print("step\tcost\tW")
for step in range(21):
	sess.run(update, feed_dict={X: x_data, Y: y_data})
	print(step, sess.run(cost, feed_dict={X: x_data, Y:y_data}), sess.run(W))

