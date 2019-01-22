import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")

sess = tf.Session()
print(sess.run(hello))

#hello라는 노드를 만들고 세션을 만들어서 hello를 실행시킴
