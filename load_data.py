import numpy as np
import tensorflow as tf
tf.set_random_seed(777)


#파일 이름, 구분 방법, 타입
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

#입력받은 2차원 배열을 X, Y데이터로 나눔
x_data = xy[:, 0:-1]	# 0 ~ n-1
y_data = xy[:, [-1]]	# n

# 입력데이터 확인
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)


# 이전과 동일
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#training
for step in range(2001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y:y_data})
	
	if step%10 == 0:
		print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)


#Prediction
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
