# import tensorflow as tf
# import numpy as np
#
# # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
# x_data = np.float32(np.random.rand(2, 100)) # 随机输入
# y_data = np.dot([0.100, 0.200], x_data) + 0.300
#
# # 构造一个线性模型
# # 构建一个简单的线性拟合模型
# # 给了一个初始值
# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b
#
# # 最小化方差
# # 构建损失函数,降低...的均值
# loss = tf.reduce_mean(tf.square(y - y_data))
# # optimizer = tf.train.AdamOptimizer()
# # 梯度下降优化器
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# # 最小化误差
# train = optimizer.minimize(loss)
#
# # 初始化变量
# # 初始化所有变量
# init = tf.initialize_all_variables()
#
# # 启动图 (graph)
# # 也饿一说是启动计算
# sess = tf.Session()
# sess.run(init)
#
# # 拟合平面
# for step in range(0, 201):
#     # sess.run就是计算指定内容
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(W), sess.run(b))
#
# # 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
'''-----------------------------------------------------矩阵乘法----------------------------------------'''
# import tensorflow as tf
#
# '''--------------------------构建---------------------'''
# # 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# # 加到默认图中.
# #
# # 构造器的返回值代表该常量 op 的返回值.
# matrix1 = tf.constant([[3., 3.]])
#
# # 创建另外一个常量 op, 产生一个 2x1 矩阵.
# matrix2 = tf.constant([[2.],[2.]])
#
# # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# # 返回值 'product' 代表矩阵乘法的结果.
# product = tf.matmul(matrix1, matrix2)
# '''------------------------执行---------------------'''
# sess = tf.Session()
# print(sess.run(product))
# sess.close()
'''-------------------------------------------------选择GPU-------------------------------------'''
# import tensorflow as tf
# with tf.Session() as sess:
#   with tf.device("/gpu:0"):
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.],[2.]])
#     product = tf.matmul(matrix1, matrix2)
#     ans = sess.run(product)
#     print(ans)
'''进入一个交互式 TensorFlow 会话.'''
# import tensorflow as tf
# sess = tf.InteractiveSession()
#
# x = tf.Variable([1.0, 2.0]) # 定义一个变量
# a = tf.constant([3.0, 3.0]) # 定义一个常量
#
# # 使用初始化器 initializer op 的 run() 方法初始化 'x'
# x.initializer.run()
#
# # 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
# sub = tf.subtract(x, a)
# print(sub.eval())
# # ==> [-2. -1.]
'''-----计算图的每一个步骤都需要run否则无法真正的运行'''
import tensorflow as tf
# 计数器
state = tf.Variable(initial_value=0,name="counter")
# 创建操作,构建计算图
one = tf.constant(value=1)
new_value = tf.add(state,one)
update_state = tf.assign(ref=state,value=new_value)
# 初始化所有变量,所有变量必须初始化
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(one),sess.run(state))