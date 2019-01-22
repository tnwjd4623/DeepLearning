import tensorflow as tf

a = tf.placeholder(tf.float32)	#a 노드
b = tf.placeholder(tf.float32)	#b 노드

adder_node = a+b #더하기 노드 생성

#a, b, adder 그래프 생성
sess = tf.Session()

print(sess.run(adder_node, feed_dict={a:3, b:4.5 } ) )
print(sess.run(adder_node, feed_dict={a: [1,3], b:[2,4] } ) )


#실행 도중에 feed_dict로 값을 넘겨줄 수 있다
