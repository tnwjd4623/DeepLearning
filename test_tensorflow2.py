import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1:", node1, "node2:", node2)		#node에 대한 정보 출력
print("node3:", node3)

sess = tf.Session()
print("sess.run(node1, node2):", sess.run([node1, node2])) #원하는 숫자 값 출력
print("sess.run(node3):", sess.run(node3))

#run안에 있는 인자를 실행시킴, 세션으로 실행시켜야 원하는 값을 얻을 수 있다
#node 1, 2, 3의 그래프를 로드하고 Session으로 실행시킴
