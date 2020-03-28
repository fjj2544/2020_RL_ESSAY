import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
# from policy_value_network.liner_sample import Sample
# from env.plane.liner_env import Planes_Env
save_model_dir = "./current_best_dynamic_fit_plane_model"
'''用一个不怎么随机的方法去采样,肯定效果不好'''
'''如果说完全不随机,不需要策略网络当前就是一个纯策略'''
class Dynamic_Net():
    def __init__(self,env, lr=0.0001, model_file=None):
        self.n_features = env.observation_dim # 输入特征维度
        self.learning_rate = lr # 学习率
        self.loss_alpha = 0.95 # 过去的loss占比
        self.batch = 100 # 批大小
        self.iter = 2000 # 迭代总次数
        self.n_actions = env.action_dim # 动作空间维度,特别现在是连续动作
        # 1.1 输入层  构建输入层 (由于是静态图,所以有placeholder)
        self.obs_action = tf.placeholder(tf.float32, shape=[None, self.n_features+self.n_actions])
        # 1.2.第一层隐含层100个神经元,激活函数为relu
        self.f1 = tf.layers.dense(inputs=self.obs_action, units=200, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1), \
                                  bias_initializer=tf.constant_initializer(0.1))
        # 1.3 第二层隐含层100个神经元，激活函数为relu
        self.f2 = tf.layers.dense(inputs=self.f1, units=100, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),\
                                  bias_initializer=tf.constant_initializer(0.1))
        # 1.4 输出层3个神经元，没有激活函数  怎么感觉这个玩意并不是所谓的学习PDE呢
        self.predict = tf.layers.dense(inputs=self.f2, units= self.n_features)
        # 2. 构建损失函数
        self.delta =  tf.placeholder(tf.float32,[None, self.n_features]) #说白了tf给出一个一种计算的流程或者说计算的方法
        self.loss = tf.reduce_mean(tf.square(self.predict-self.delta))
        # 3. 定义一个优化器
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # 4. tf工程
        self.sess = tf.Session() # 图构建完毕之后就可以创建tf工程
        # 5. 初始化图中的变量,类似于C++中的实例化
        self.sess.run(tf.global_variables_initializer())
        # 6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        # 迭代训练
        if model_file is not None:
            self.restore_model(model_file)
    # 拟合动力学方程,采样获得正确标签拟合动力学模型
    def fit_dynamic(self,batch_obs_act,batch_delta,save_dir = None,epoch =10):
        flag = 0
        train_obs_act = batch_obs_act
        train_delta = batch_delta
        N = train_delta.shape[0]
        train_indicies = np.arange(N) # 获得时序标签
        loss_line=[]
        num = 0
        ls = 0
        #训练神经网络
        for i in range(self.iter):
            np.random.shuffle(train_indicies)
            for j in range(int(math.ceil(N/self.batch))):
                start_idx = j * self.batch%N
                idx = train_indicies[start_idx:start_idx+self.batch] # 获取一批数据的索引
                # 喂数据训练
                self.sess.run([self.train_op], feed_dict={self.obs_action:train_obs_act[idx,:], self.delta:train_delta[idx,:]})
                loss = self.sess.run([self.loss],feed_dict={self.obs_action:train_obs_act[idx,:], self.delta:train_delta[idx,:]})
                loss_line.append(loss)
                # print(loss[0]) # 这个主要是为了把数据提取出来
                if num == 0:
                    ls=loss[0]
                else:
                    ls = self.loss_alpha*ls+(1-self.loss_alpha)*loss[0]
                num+=1
                if i > epoch:
                    flag=1
                    break
            print("第%d次实验,误差为%f" % (i, ls))
            if flag == 1:
                break
        if save_dir is not None:
            self.save_model(save_dir)
        number_line=np.arange(len(loss_line)) # 获得训练次数的标签
        plt.plot(number_line, loss_line)
        plt.show()
    def prediction(self,s_a, target_state = None,if_debug = False):
        #正则化数据
        norm_s_a = s_a
        #利用神经网络进行预测
        delta = self.sess.run(self.predict, feed_dict={self.obs_action:norm_s_a})
        predict_out = delta + s_a[:,0:self.n_features]
        '''Debug'''
        if if_debug and target_state is not None:
            x = np.arange(len(predict_out))
            plt.figure(1)
            plt.plot(x, predict_out[:,0],label="predict")
            plt.plot(x, target_state[:,0],'--',label="real")
            plt.legend(loc="best")
            # plt.savefig("./alpha.png")
            plt.figure(2)
            plt.plot(x, predict_out[:, 1], label="predict")
            plt.plot(x, target_state[:, 1], '--', label="real")
            plt.legend(loc="best")
            # plt.savefig("./theta.png")
            plt.figure(3)
            plt.plot(x, predict_out[:, 2], label="predict")
            plt.plot(x, target_state[:, 2], '--', label="real")
            plt.legend(loc="best")
            # plt.savefig("./q.png")
            plt.show()
        return predict_out
    # 定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
# '''DEBUG'''
# env = Planes_Env()
# sampler = Sample(env)
# dynamic_fit_network = Dynamic_Net(env,sampler)
# sigma = [1,0.5,0.1]
# mu = [1.5,2.5,0.5]
#
# '''训练集 这里需要完成一个多次训练拼接的任务'''
# # train_x = None # 相当于训练集是空的
# # train_y = None
# # real_y = None
# # if_experince_empty = True# 经验池是否为空
# # for i in range(100):
# #     K = np.random.normal(mu,sigma)
# #     x,y,r_y = sampler.sample_episodes_with_PID(K,1)
# #     # 第一次采样直接赋值
# #     if if_experince_empty:
# #         train_x = x
# #         train_y = y
# #         real_y = r_y
# #         if_experince_empty = False
# #     else:
# #         train_x = np.vstack((train_x,x))
# #         train_y = np.vstack((train_y,y))
# #         real_y = np.vstack((real_y,r_y))
# # print(train_x.shape,train_y.shape,real_y.shape)
# # dynamic_fit_network.fit_dynamic(train_x,train_y,save_model_dir)
# '''测试集'''
# K = np.random.normal(mu,sigma)
# train_x , train_y , real_y = sampler.sample_episodes_with_PID(K,1)
# dynamic_fit_network.prediction(train_x,real_y,True)


