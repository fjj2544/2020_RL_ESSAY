import pygame
import numpy as np
import tensorflow as tf
# from load import *
from pygame.locals import *
import math
import time
import matplotlib.pyplot as plt
RENDER = False
import random
C_UPDATE_STEPS = 1
A_UPDATE_STEPS = 1
# 高超声速飞行器定点飞行，高度33.5km，速度15Ma
# alpha 飞机迎角
# theta 飞机俯仰角
# q 飞机俯仰角速度
# delta_z 飞机俯仰舵偏角
# b_alpha = 0.073
# b_deltaz = -0.0035
# a_alpha = 0.7346
# a_deltaz = -2.8375
# a_q = 3.9779
# dq = -a_alpha*alpha - a_q*q - a_delta_z*delta_z
# dtheta = q
# dalpha = q - b_alpha*alpha -b_delta_z*delta_z
#定义环境
class Planes_Env:
    def __init__(self):
        # self.actions = [0,1]
        self.observation_dim = 2
        # 一阶变量
        self.altitude = 33500.0
        self.Mach = 15.0
        self.theta = 0.0
        self.pitch = 0.0 / 57.3
        self.rrange = 0.0
        self.mass = 83191
        self.omega_z = 0.0
        # 速度
        self.daltitude = 0.0
        self.dMach = 0.0
        self.dtheta = 0.0
        self.dpitch = 0.0
        self.drrange = 0.0
        self.dmass = 0.0
        self.domega_z = 0.0
        # 攻角
        self.arfa = 0.0 / 57.3
        # 目标
        self.pitch_desired = 5 / 57.3
        self.dpithch_desired = 0.0
        self.theta_desired = 0.0
        self.dtheta_desired = 0.0
        #其他常量
        self.Vs = 305.58
        self.Lr =24.38
        self.G0 =9.81
        self.Sr = 334.73
        self.Jz = 8288700
        self.Re = 6371000
        #状态量分别为[飞机迎角 alpha, 飞机俯仰角theta, 飞机俯仰角速度q]
        # self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.observation = np.array([0.0, 0.0])
        self.state = np.array([0.0,0.0])
        self.steps_beyond_done = 0
        self.max_steps = 400
        self.viewer = None
        #角度阈值 攻角alpha[-1,10];舵偏delta_z[-20,20]
        self.alpha_threshold_max = 10
        self.alpha_threshold_min = -1
        self.delta_z_threhold_max = 20
        self.delta_z_threhold_min = -20
        self.reward = 0.0
        self.cost = 0.0
        self.tau = 0.005
    def reset(self):
        n = np.random.randint(1,1000,1)
        np.random.seed(n)
        # self.state = np.random.uniform(-0.5,0.5 ,size=(7,) )
        # self.state = np.array([0.0,0.0,0.0])
        self.observation = np.array([0.0, 0.0])
        #一阶变量
        self.altitude = 33500.0
        self.Mach = 15.0
        self.theta = 0.0
        self.pitch = 0.0 / 57.3
        self.rrange = 0.0
        self.mass = 83191
        self.omega_z = 0.0
        #速度
        self.daltitude = 0.0
        self.dMach = 0.0
        self.dtheta = 0.0
        self.dpitch = 0.0
        self.drrange = 0.0
        self.dmass = 0.0
        self.domega_z = 0.0
        #攻角
        self.arfa = 0.0 / 57.3
        #目标
        self.pitch_desired = 5 / 57.3
        self.dpithch_desired = 0.0
        self.theta_desired = 0.0
        self.dtheta_desired = 0.0
        # self.state = np.array([0.0, 5.0, 0.0])
        self.steps_beyond_done = 0
        # print(self.state)
        return self.observation
    def step(self,action):
        # print("action", action)
        action = action[0]
        Alpha_deg = self.arfa*57.3
        # state = self.state
        Rho = np.exp(-2.114 * 10.0 ** (-14.0) * self.altitude ** 3.0 + 3.656 * 10.0 ** (
            -9.0) * self.altitude ** 2.0 - 3.309 * 10.0 ** (-4.0) * self.altitude + 3.217)
        Qdyn = 0.5 * Rho * self.Mach * self.Vs * self.Mach * self.Vs
        # ***************************** 高超声速 	升力系数 ********************
        CL0 = -8.19 * 10.0 ** (-2) + 4.70 * 10.0 ** (-2) * self.Mach + 1.86 * 10.0 ** (-2) * Alpha_deg \
              - 4.73 * 10.0 ** (-4) * (Alpha_deg * self.Mach) - 9.19 * 10.0 ** (-3) * self.Mach ** 2 - 1.52 * 10.0 ** (
                  -4) * Alpha_deg ** 2 \
              + 7.74 * 10.0 ** (-4) * self.Mach ** 3 + 5.99 * 10.0 ** (-7) * (Alpha_deg * self.Mach) ** 2 \
              + 4.08 * 10.0 ** (-6) * Alpha_deg ** 3 - 2.93 * 10.0 ** (-5) * self.Mach ** 4 - 3.91 * 10.0 ** (
                  -7) * Alpha_deg ** 4 \
              + 4.12 * 10.0 ** (-7) * self.Mach ** 5 + 1.30 * 10.0 ** (-8) * Alpha_deg ** 5
        CL_e = -1.45 * 10.0 ** (-5) + 1.01 * 10.0 ** (-4) * Alpha_deg + 7.10 * 10.0 ** (-6) * self.Mach \
               - 4.14 * 10.0 ** (-4) * action - 3.51 * 10.0 ** (
                   -6) * Alpha_deg * action + 8.72 * 10.0 ** (-6) * self.Mach * action \
               + 1.70 * 10.0 ** (-7) * self.Mach * Alpha_deg * action
        CL_a = CL_e
        # ***************************** 高超声速 阻力系数 ********************
        CD0 = 8.717 * 10.0 ** (-2) - 3.307 * 10.0 ** (-2) * self.Mach + 3.179 * 10.0 ** (-3) * Alpha_deg \
              - 1.250 * 10.0 ** (-4) * (Alpha_deg * self.Mach) + 5.036 * 10.0 ** (-3) * self.Mach ** 2 \
              - 1.100 * 10.0 ** (-3) * Alpha_deg ** 2 + 1.405 * 10.0 ** (-7) * (Alpha_deg * self.Mach) ** 2 \
              - 3.658 * 10.0 ** (-4) * self.Mach ** 3 + 3.175 * 10.0 ** (-4) * Alpha_deg ** 3 + 1.274 * 10.0 ** (
                  -5) * self.Mach ** 4 \
              - 2.985 * 10.0 ** (-5) * Alpha_deg ** 4 - 1.705 * 10.0 ** (-7) * self.Mach ** 5 + 9.766 * 10.0 ** (
                  -7) * Alpha_deg ** 5
        CD_e = 4.5548 * 10.0 ** (-4) + 2.5411 * 10.0 ** (-5) * Alpha_deg - 1.1436 * 10.0 ** (-4) * self.Mach \
               + 3.2187 * 10.0 ** (-6) * Alpha_deg ** 2 + 3.014 * 10.0 ** (-6) * self.Mach ** 2 \
               - 3.6417 * 10.0 ** (-5) * action - 5.3015 * 10.0 ** (-7) * self.Mach * Alpha_deg * action \
               + 6.9629 * 10.0 ** (-6) * action ** 2 + 2.1026 * 10.0 ** (-12) * (
                       self.Mach * Alpha_deg * action) ** 2
        CD_a = CD_e
        CD_r = 7.50 * 10.0 ** (-4) - 2.29 * 10.0 ** (-5) * Alpha_deg - 9.69 * 10.0 ** (-5) * self.Mach + 8.76 * 10.0 ** (
            -7) * Alpha_deg ** 2 + 2.70 * 10.0 ** (-6) * self.Mach ** 2
        CD = CD0 + CD_e + CD_a + CD_r
        CL = CL0 + CL_e + CL_a
        # ***************************** 高超声速 俯仰力矩 ********************
        mz0 = -2.192 * 10.0 ** (-2) + 7.739 * 10.0 ** (-3) * self.Mach - 2.260 * 10.0 ** (-3) * Alpha_deg \
              + 1.808 * 10.0 ** (-4) * (Alpha_deg * self.Mach) - 8.849 * 10.0 ** (-4) * self.Mach ** 2 \
              + 2.616 * 10.0 ** (-4) * Alpha_deg ** 2 - 2.880 * 10.0 ** (-7) * (Alpha_deg * self.Mach) ** 2 \
              + 4.617 * 10.0 ** (-5) * self.Mach ** 3 - 7.887 * 10.0 ** (-5) * Alpha_deg ** 3 - 1.143 * 10.0 ** (
                  -6) * self.Mach ** 4 \
              + 8.288 * 10.0 ** (-6) * Alpha_deg ** 4 + 1.082 * 10.0 ** (-8) * self.Mach ** 5 - 2.789 * 10.0 ** (
                  -7) * Alpha_deg ** 5
        mz_e = -5.67 * 10.0 ** (-5) - 1.51 * 10.0 ** (-6) * self.Mach - 6.59 * 10.0 ** (
            -5) * Alpha_deg + 2.89 * 10.0 ** (-4) * action \
               + 4.48 * 10.0 ** (-6) * Alpha_deg * action - 4.46 * 10.0 ** (
                   -6) * self.Mach * Alpha_deg - 5.87 * 10.0 ** (-6) * self.Mach * action \
               + 9.72 * 10.0 ** (-8) * self.Mach * Alpha_deg * action
        mz_a = mz_e
        mz_r = -2.79 * 10.0 ** (-5) * Alpha_deg - 5.89 * 10.0 ** (-8) * Alpha_deg ** 2 + 1.58 * 10.0 ** (-3) * self.Mach ** 2 \
               + 6.42 * 10.0 ** (-8) * Alpha_deg ** 3 - 6.69 * 10.0 ** (-4) * self.Mach ** 3 \
               - 2.10 * 10.0 ** (-8) * Alpha_deg ** 4 + 1.05 * 10.0 ** (-4) * self.Mach ** 4 \
               + 3.14 * 10.0 ** (-9) * Alpha_deg ** 5 - 7.74 * 10.0 ** (-6) * self.Mach ** 5 \
               - 2.18 * 10.0 ** (-10) * Alpha_deg ** 6 + 2.70 * 10.0 ** (-7) * self.Mach ** 6 \
               + 5.74 * 10.0 ** (-12) * Alpha_deg ** 7 - 3.58 * 10.0 ** (-9) * self.Mach ** 7
        mzz = -1.36 + 0.386 * self.Mach + 7.85 * 10.0 ** (-4) * Alpha_deg + 1.40 * 10.0 ** (-4) * Alpha_deg * self.Mach \
              - 5.42 * 10.0 ** (-2) * self.Mach ** 2 + 2.36 * 10.0 ** (-3) * Alpha_deg ** 2 - 1.95 * 10.0 ** (-6) * (
                      Alpha_deg * self.Mach) ** 2 \
              + 3.80 * 10.0 ** (-3) * self.Mach ** 3 - 1.48 * 10.0 ** (-3) * Alpha_deg ** 3 \
              - 1.30 * 10.0 ** (-4) * self.Mach ** 4 + 1.69 * 10.0 ** (-4) * Alpha_deg ** 4 \
              + 1.71 * 10.0 ** (-6) * self.Mach ** 5 - 5.93 * 10.0 ** (-6) * Alpha_deg ** 5
        mz = mz0 + mz_e + mz_a + mz_r + mzz * self.omega_z * self.Lr / (2 * self.Mach * self.Vs)
        Lift = Qdyn * CL * self.Sr
        Drag = Qdyn * CD * self.Sr
        Mz = Qdyn * mz * self.Sr * self.Lr
        Thrust = 1.9 * 10.0 ** 5.0
        Isp = 4900
        # 动力学方程
        self.daltitude = self.Mach * self.Vs * np.sin(self.theta)
        self.dMach = (Thrust * np.cos(self.arfa) - Drag - self.mass * self.G0 * np.sin(self.theta)) / (self.mass *self.Vs)
        self.dtheta = (Thrust * np.sin(self.arfa) + Lift) / (self.mass * self.Mach * self.Vs) + np.cos(self.theta) * (
                    self.Mach * self.Vs / (self.Re + self.altitude) - self.G0 / (self.Mach * self.Vs))
        self.dmass = -Thrust / (self.G0 * Isp)
        self.drrange = self.Mach * self.Vs * np.cos(self.theta) * (self.Re / (self.Re + self.altitude))
        self.domega_z = Mz / self.Jz
        self.dpitch = self.omega_z

        self.altitude = self.altitude + self.daltitude * self.tau
        self.Mach = self.Mach+self.dMach * self.tau
        self.theta = self.theta + self.dtheta * self.tau
        self.rrange = self.rrange + self.drrange * self.tau
        self.mass = self.mass + self.dmass * self.tau
        self.omega_z = self.omega_z + self.domega_z * self.tau
        self.pitch = self.pitch + self.dpitch * self.tau
        self.arfa = self.pitch - self.theta
        self.steps_beyond_done += 1
        #根据更新的状态判断是否结束
        lose =  Alpha_deg < self.alpha_threshold_min or Alpha_deg > self.alpha_threshold_max
        #设置回报
        if not lose :
            self.reward =-((self.pitch*57.3-self.pitch_desired*57.3)**2+0.1*(self.dpitch*57.3-self.dpitch*57.3)**2+0.01*action**2)
        else:
            self.reward = -4500
        done = lose or self.steps_beyond_done > self.max_steps
        self.observation = np.array([self.pitch*57.3, self.dpitch*57.3])
        return self.observation, self.reward, done


class Experience_Buffer():
    def __init__(self,buffer_size = 5000):
        self.buffer = []
        self.buffer_size = buffer_size
    def add_experience(self,experience):
        if len(self.buffer)+len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size]=[]
        self.buffer.extend(experience)
    def sample(self, samples_num):
        sample_data = np.reshape(np.array(random.sample(self.buffer, samples_num)),[samples_num, 4])
        train_obs = np.array(sample_data[0,0])
        train_obs_ = np.array(sample_data[0,3])
        train_a = sample_data[:, 1]
        train_r = sample_data[:, 2]
        for i in range(samples_num-1):
            train_obs = np.vstack((train_obs, np.array(sample_data[i+1,0])))
            train_obs_ = np.vstack((train_obs_, np.array(sample_data[i+1,3])))
        train_obs = np.reshape(train_obs,[samples_num,2])
        train_obs_ = np.reshape(train_obs_,[samples_num,2])
        train_r = np.reshape(train_r, [samples_num,1])
        train_a = np.reshape(train_a,[samples_num,1])
        return train_obs, train_a, train_r, train_obs_
#定义策略网络
class Policy_Net():
    def __init__(self, env, action_bound, lr = 0.001, model_file=None):
        self.action_bound = action_bound
        self.gamma = 0.95
        self.tau = 0.01
        #  tf工程
        self.sess = tf.Session()
        self.learning_rate = lr
        #输入特征的维数
        self.n_features = env.observation_dim
        #输出动作空间的维数
        self.n_actions = 1
        #1. 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.obs_ = tf.placeholder(tf.float32, shape=[None, self.n_features])
        #2.创建网络模型
        #2.1 创建策略网络，策略网络的命名空间为: 'actor'
        with tf.variable_scope('actor'):
            #可训练的策略网络,可训练的网络参数命名空间为: actor/eval:
            self.action = self.build_a_net(self.obs, scope='eval', trainable=True)
            #靶子策略网络，不可训练,网络参数命名空间为：actor/target:
            self.action_=self.build_a_net(self.obs_, scope='target',trainable=False)
        #2.2 创建行为值函数网络，行为值函数的命名空间为: 'critic'
        with tf.variable_scope('critic'):
            #可训练的行为值网络，可训练的网络参数命名空间为:critic/eval
            Q = self.build_c_net(self.obs, self.action, scope='eval', trainable=True)
            Q_ = self.build_c_net(self.obs_, self.action_, scope='target', trainable=False)
        #2.3 整理4套网络参数
        #2.3.1：可训练的策略网络参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
        #2.3.2: 不可训练的策略网络参数
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        #2.3.3: 可训练的行为值网络参数
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/eval')
        #2.3.4: 不可训练的行为值网络参数
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target')
        #2.4 定义新旧参数的替换操作
        self.update_olda_op = [olda.assign((1-self.tau)*olda+self.tau*p) for p,olda in zip(self.ae_params, self.at_params)]
        self.update_oldc_op = [oldc.assign((1-self.tau)*oldc+self.tau*p) for p,oldc in zip(self.ce_params, self.ct_params)]
        #3.构建损失函数
        #3.1 构建行为值函数的损失函数
        self.R = tf.placeholder(tf.float32, [None, 1])
        Q_target = self.R + self.gamma * Q_
        self.c_loss = tf.losses.mean_squared_error(labels=Q_target, predictions=Q)
        #3.2 构建策略损失函数，该函数为行为值函数
        self.a_loss=-tf.reduce_mean(Q)
        #4. 定义优化器
        #4.1 定义动作优化器,注意优化的变量在ca_params中
        self.a_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss, var_list=self.ae_params)
        #4.2 定义值函数优化器，注意优化的变量在ce_params中
        self.c_train_op = tf.train.AdamOptimizer(0.002).minimize(self.c_loss, var_list=self.ce_params)
        #5. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    def build_c_net(self,obs, action, scope, trainable):
        with tf.variable_scope(scope):
            c_l1 = 50
            #与状态相对应的权值
            w1_obs = tf.get_variable('w1_obs',[self.n_features, c_l1], trainable=trainable)
            #与动作相对应的权值
            w1_action = tf.get_variable('w1_action',[self.n_actions, c_l1],trainable=trainable)
            b1 = tf.get_variable('b1',[1, c_l1], trainable=trainable)
            #第一层隐含层
            c_f1 = tf.nn.relu(tf.matmul(obs, w1_obs)+tf.matmul(action,w1_action)+b1)
            # 第二层， 行为值函数输出层
            c_out = tf.layers.dense(c_f1, units=1, trainable=trainable)
        return c_out
    def build_a_net(self, obs, scope, trainable):
        with tf.variable_scope(scope):
            # 行为值网络第一层隐含层
            a_f1 = tf.layers.dense(inputs=obs, units=400, activation=tf.nn.relu, trainable=trainable)
            # 第二层， 确定性策略
            a_out = 20*tf.layers.dense(a_f1, units=self.n_actions,activation=tf.nn.tanh, trainable=trainable)
            return tf.clip_by_value(a_out, action_bound[0], action_bound[1])
    #根据策略网络选择动作
    def choose_action(self, obs):
        action = self.sess.run(self.action, {self.obs:obs})
        # print("greedy action",action)
        # print(action[0])
        return action[0]
    #定义训练
    def train_step(self, train_obs, train_a, train_r, train_obs_):
        for _ in range(A_UPDATE_STEPS):
            self.sess.run(self.a_train_op, feed_dict={self.obs:train_obs})
        for _ in range(C_UPDATE_STEPS):
            self.sess.run(self.c_train_op, feed_dict={self.obs:train_obs, self.action:train_a, self.R:train_r, self.obs_:train_obs_})
        # 更新旧的策略网络
        self.sess.run(self.update_oldc_op)
        self.sess.run(self.update_olda_op)
        # return a_loss, c_loss
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
def policy_train(env, brain, exp_buffer, training_num):
    reward_sum = 0
    average_reward_line = []
    training_time = []
    average_reward = 0
    max_rewards = 0.0
    batch = 320
    # for i in range(training_num):
    #     sample_states,sample_actions, sample_rs = sample.sample_steps(32)
    #     a_loss,c_loss = brain.train_step(sample_states, sample_actions,sample_rs)
    for i in range(training_num):
        total_reward = 0
        #初始化环境
        observation = env.reset()
        done = False
        while True:
            #探索权重衰减
            var = 3*np.exp(-i/2000)
            observation = np.reshape(observation, [1,brain.n_features])
            #根据神经网络选取动作
            action = brain.choose_action(observation)
            #给动作添加随机项，以便进行探索
            action = np.clip(np.random.normal(action, var), -20, 20)
            obeservation_next, reward, done= env.step(action)
            # 存储一条经验
            experience = np.reshape(np.array([observation,action[0],reward,obeservation_next]),[1,4])
            exp_buffer.add_experience(experience)
            if len(exp_buffer.buffer)>batch:
                #采样数据，并进行训练
                train_obs, train_a, train_r, train_obs_ = exp_buffer.sample(batch)
                #学习一步
                brain.train_step(train_obs, train_a, train_r, train_obs_)
            #推进一步
            observation = obeservation_next
            total_reward += reward
            if done:
                break
        if i == 0:
            average_reward = total_reward
            max_rewards = average_reward
        else:
            average_reward = 0.95*average_reward + 0.05*total_reward
        if average_reward > max_rewards:
            max_rewards = average_reward
            brain.save_model('./current_best_ddpg_plane_nonlinear_2d')
        print("第%d次学习后的平均回报为：%f,最大回报为%f"%(i,average_reward,max_rewards))
        average_reward_line.append(average_reward)
        training_time.append(i)
        if average_reward > -500:
            break
    # brain.save_model('./current_best_ddpg_pendulum')
    plt.plot(training_time, average_reward_line)
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()
#测试当前的网络, 跟单摆不同的地方
def policy_test(env, policy,RENDER,test_number):
    reward_sum = 0
    time_steps=[]
    pitch_thetas = []
    alphas=[]
    qs = []
    steps = 0
    actions = []
    for i in range(test_number):
        observation = env.reset()
        while True:
            time_steps.append(steps)
            pitch_thetas.append(env.pitch)
            alphas.append(env.arfa)
            qs.append(env.dpitch)
            #根据策略网络产生一个动作
            action = policy.choose_action(np.reshape(observation,[1,2]))
            actions.append(action[0])
            observation_, reward, done = env.step(action)
            if RENDER:
                print('reward',reward)
            reward_sum += reward
            steps+=1
            if done:
                break
            observation = observation_
    if RENDER:
        print("pitch_thetas",pitch_thetas)
        print("actions", actions)
        plt.figure(1)
        plt.plot(time_steps, pitch_thetas)
        plt.xlabel("time_steps")
        plt.ylabel("theta")
        plt.figure(2)
        plt.plot(time_steps, alphas)
        plt.xlabel("time_steps")
        plt.ylabel("alpha")
        plt.figure(3)
        plt.plot(time_steps, qs)
        plt.xlabel("time_steps")
        plt.ylabel("qs")
        plt.figure(4)
        plt.plot(time_steps,actions)
        plt.xlabel("time_steps")
        plt.ylabel("actions")
        plt.show()
    # print(reward_sum)
    return reward_sum/test_number

if __name__=="__main__":
    env = Planes_Env()
    #定义舵角的取值范围[-20,20]
    action_bound = [-20, 20]
    #实例化一个 策略网络
    brain = Policy_Net(env, action_bound)
    # 经验缓存
    exp_buffer = Experience_Buffer()
    training_num =1000
    #训练策略网络
    policy_train(env, brain,exp_buffer,training_num)
    policy_test(env,brain,True,1)







