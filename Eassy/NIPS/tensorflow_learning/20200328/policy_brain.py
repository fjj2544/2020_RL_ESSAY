import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from  liner_env import  Planes_Env

class Policy_Net():
    # 初始化网络
    def __init__(self, observation_dim, action_dim, policy_name, lr=1e-5, model_file=None):
        self.n_features = observation_dim       # 输入特征维度
        self.n_actions = action_dim             # 动作空间维度,特别现在是连续动作
        self.learning_rate = lr                     # 学习率
        self.loss_alpha = 0.95                      # 过去的loss占比
        self.action_bound = [-20, 20]               # 动作限幅
        # 采样参数
        self.sigma_exploration = tf.placeholder(tf.float32, shape=[None, 1])
        # >>>>>>>>>构建网络>>>>>>>>>
        # 1:    输入层
        with tf.variable_scope(policy_name):
            self.input_layer = tf.placeholder(tf.float32, shape=[None, self.n_features])
            # 2:    100神经元&relu
            self.policy_1 = tf.layers.dense(inputs=self.input_layer, units=100, activation=tf.nn.relu, trainable=True)
            # 3:    60神经元&relu
            self.policy_2 = tf.layers.dense(inputs=self.policy_1, units=60, activation=tf.nn.relu, trainable=True)
            # 动作
            self.action = 20 * tf.layers.dense(inputs=self.policy_2, units=self.n_actions, activation=tf.nn.tanh, trainable=True)
            # 网络参数
            self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=policy_name)
        # 采样
        self.normal_dist = tf.contrib.distributions.Normal(self.action, self.sigma_exploration)
        self.sample = tf.clip_by_value(tf.squeeze(self.normal_dist.sample(1),axis=0), self.action_bound[0], self.action_bound[1])
        self.target_act = tf.placeholder(tf.float32, [None, 1])
        # 当前的值函数
        self.loss = tf.reduce_mean(tf.square(self.target_act - self.action))
        # 优化器
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # tf工程
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())
        # 6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        # 模型保留
        if model_file is not None:
            self.restore_model(model_file)
    # 采样动作
    def policy_sample(self, sigma, observation):
        action = self.sess.run([self.sample], feed_dict={self.input_layer: observation, self.sigma_exploration: sigma})
        return action
    # 实际动作
    def policy(self, observation):
        action = self.sess.run([self.action], feed_dict={self.input_layer: observation})
        return action
    # 学习新的知识
    def learn(self, batch_obs, batch_target_act ,epoch = 10):
        for i in range(epoch):
            self.sess.run([self.train_op], feed_dict={self.input_layer: batch_obs, self.target_act: batch_target_act})
            loss = self.sess.run([self.loss], feed_dict={self.input_layer: batch_obs, self.target_act: batch_target_act})
        # print(loss)
        # print(loss)
        return float(loss[0])
    # 保存网络
    def save_model(self, model_path="./agent_model/policy_net"):
        self.saver.save(self.sess, model_path)
    # 重建网络
    def restore_model(self, model_path="./agent_model/policy_net"):
        self.saver.restore(self.sess, model_path)
# 感觉看样子可以学
# if __name__ == '__main__':
#     env = Planes_Env()
#     net1 = Policy_Net(env,"main_policy")
#     obs = np.ones([3,3])
#     # a = net1.policy(obs)
#     # a = np.reshape(a,newshape=[len(obs),sim_env.action_dim])
#     # net1.learn(obs,a)
#     '''测试policy_sample'''
#     sigma = np.ones([3,1])
#     a2 = net1.policy_sample(sigma,obs)
#     print(a2)