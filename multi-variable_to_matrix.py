import tensorflow as tf

#Data Set by Matrix
x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]

y_data = [[152.], [185.], [180.], [196.], [142.]]


X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

#weight and bias
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#H(x) = Y 가 목표 / Matrix Multiplication
hypothesis = tf.matmul(X, W) + b

#cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


#train
#cost, hypothesis 출력, x1,x2,x3에 값 배정

#hypothesis = y [...] 의 값에 가까워져야 성공
for step in range(2001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y:y_data})

	if step%10 == 0:
		print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


