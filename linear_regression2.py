import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name="weight") # tensorflow가 사용하는 변수라고 생각
b = tf.Variable(tf.random_normal([1]), name="bias") 
# W와 b의 값을 모르기 때문에 rank가 1인 random 값

hypothesis = X * W + b		# H(x) = Wx + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost(W, b) = H(x) - y
#reduce_mean 은 값들의 평균을 내주는 함수


#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)	#minimize의 인자(cost)를 최소화하도록 학습
# 즉 W와 b값을 바꾸면서 최소 값을 찾음

#---------- 여기 까지 그래프 구현 단계 -----------

sess = tf.Session()
sess.run(tf.global_variables_initializer())  # tf변수 사용 전에는 반드시 실행시켜주어야함


# 20번에 한번씩 결과 출력 cost, W, b출력
for step in range(2001):
	cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1,2,3,4,5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
	if step % 20 == 0:
		print(step, cost_val, W_val, b_val)

# W = 1, b = 0으로 수렴할 때 , cost는 작은 값

