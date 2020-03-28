import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
W = tf.get_variable("w", shape=[2, 1], initializer=tf.constant_initializer([0.1, 0.2]))
b = tf.get_variable("b", shape=[1], initializer=tf.constant_initializer(0.1))
hypothesis = tf.matmul(X, W) + b
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
tf.add_to_collection('X', X) # 输入
tf.add_to_collection('Y', Y) # 标签
tf.add_to_collection('hypothesis', hypothesis) # 中间的运算结果

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './model/test')  #保存模型
