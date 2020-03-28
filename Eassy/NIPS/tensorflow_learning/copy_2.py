import tensorflow as tf

sess = tf.Session()
grah_saver = tf.train.import_meta_graph('./model/test.meta')  # 加载模型的grah
grah_saver.restore(sess, './model/test')  # 加载模型中各种变量的值，注意这里不用文件的后缀
X = tf.get_collection("X")[0]
Y = tf.get_collection("Y")[0]
hypothesis = tf.get_collection("hypothesis")[0]
b = tf.get_variable("b1", shape=[1, 1], initializer=tf.constant_initializer(0.0))
new_layers = hypothesis + b
cost = -tf.reduce_mean(Y * tf.log(new_layers) + (1 - Y) * tf.log(1 - new_layers))
sess.run(tf.global_variables_initializer())
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
for _ in range(10):
    _, loss = sess.run([optm, cost], feed_dict={X: [[1, 2]], Y: [[1]]})
    print(loss)
