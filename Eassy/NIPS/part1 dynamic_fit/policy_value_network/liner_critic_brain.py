import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

###############################################################################################
# NOTE1:Now the V(x) of the network is the V(x) of dynamic windows, which the initial state is x.
#       It is set like this: the X_final is the x(t+tolling_step) and V(X_final) is the reward of
#       X_final?.
# NOTE2:There is a problem. What is final state? We can not make sure what is the real final state
#       because the target of the task is not the same. If we use dynamic windows, what is the
#       final state? The state of last time? But this state is not the same in different trajectory.
#       We can not make sure the final state, so we can not know where the start of V to calculate.
###############################################################################################
#                   Critic网络
#   此网络结构如下：
#       第一层：obs_action——size(状态+动作)
#       第二层：f1——100 & relu
#       输出层：reward——1
#       目标函数：reward_target——1
#       损失函数：c_loss——mean(error^2)
#   功能
#       fit_dynamic(self, batch_obs_act, batch_delta, save_dir=None, epoch =10)
#       输出：无
#       参数：进行环境模拟网络的学习：
#               状态动作：batch_obs_act——[size(状态+动作)*N]
#               状态微分：batch_delta——[size(状态)*N]
#               模型存储路径：save_dir
#               训练次数：epoch
#       restore_model(self, model_path)
#       输出：无
#       功能：存储模型
#               存储路径：model_path
#       restore_model(self, model_path)
#       输出：无
#       功能：读取模型
#               模型路径：model_path
#       prediction(self,s_a, target_state = None,if_debug = False)
#       输出：下一个状态
#       功能：仿真预测
#               正则化状态动作：s_a
#               绘图时的目标函数：target_state
#               是否绘图：if_debug
###############################################################################################

save_model_dir = "./current_best_dynamic_fit_plane_model"
# 迭代次数
ITER_CRITIC = 2000
# 批大小
BATCHSIZE_CRITIC = 100


class Critic_Net():
    # 初始化网络
    def __init__(self, env, lr=0.0001, model_file=None):
        self.draw_time = 0
        self.n_features = env.observation_dim       # 输入特征维度
        self.learning_rate = lr                     # 学习率
        self.loss_alpha = 0.95                      # 过去的loss占比
        self.batch = BATCHSIZE_CRITIC               # 批大小
        self.iter = ITER_CRITIC                     # 迭代总次数
        self.n_actions = env.action_dim             # 动作空间维度,特别现在是连续动作
        # 1.1 输入层  构建输入层
        self.obs_action = tf.placeholder(tf.float32, shape=[None, self.n_features+self.n_actions])
        # 1.2 隐含层   100个神经元 + relu
        self.c_f1 = tf.layers.dense(inputs=self.obs_action, units=100, activation=tf.nn.relu)
        # 1.3 输出层   1个神经元
        self.v = tf.layers.dense(inputs=self.c_f1, units=1)
        # 定义critic网络的损失函数,输入为td目标
        self.mc_target = tf.placeholder(tf.float32, [None,1])
        self.c_loss = tf.reduce_mean(tf.square(self.mc_target-self.v))  # 使用(G-V)^2作为损失函数
        self.c_train_op = tf.train.AdamOptimizer(0.0002).minimize(self.c_loss)
        # tf工程
        self.sess = tf.Session()    # 图构建完毕之后就可以创建tf工程
        # 初始化图中的变量,类似于C++中的实例化
        self.sess.run(tf.global_variables_initializer())
        # 6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        # 模型保留
        if model_file is not None:
            self.restore_model(model_file)

    # 模型训练
    def fit_critic(self, batch_obs_act, batch_reward, save_dir=None, epoch =10):
        flag = 0
        train_obs_act = batch_obs_act
        train_reward = batch_reward
        # Update the reward to the
        # 采样得到的数据量
        N = train_reward.shape[0]
        # 获得对应数据的标签
        train_indicies = np.arange(N)
        loss_line = []  # 损失记录
        num = 0         # 评估次数
        ls = 0          # 评估损失
        for i in range(self.iter):
            # 打乱顺序
            np.random.shuffle(train_indicies)
            # >>>>>>>>>>训练神经网络>>>>>>>>>>
            for j in range(int(math.ceil(N/self.batch))):
                # 得到训练样本的索引
                start_idx = j * self.batch%N
                idx = train_indicies[start_idx:start_idx+self.batch]
                # 对模型网络进行训练
                self.sess.run([self.c_train_op], feed_dict={self.obs_action:train_obs_act[idx,:], self.mc_target:train_reward[idx,:]})
                # 计算损失
                loss = self.sess.run([self.c_loss], feed_dict={self.obs_action:train_obs_act[idx,:], self.mc_target:train_reward[idx,:]})
                loss_line.append(loss)
                # 对当前学习情况进行评估
                if num == 0:
                    ls = loss[0]
                else:
                    ls = self.loss_alpha*ls+(1-self.loss_alpha)*loss[0]
                num += 1
                if i > epoch:
                    flag=1
                    break
            print("第%d次实验,误差为%f" % (i, ls))
            if flag == 1:
                break
            # <<<<<<<<<<训练神经网络<<<<<<<<<<
        # 保存模型
        if save_dir is not None:
            self.save_model(save_dir)
        # 绘制损失变化图
        # 绘制损失变化图
        print("==============DRAW the picture============")
        # 对比计算情况
        reward = self.sess.run([self.v], feed_dict={self.obs_action: train_obs_act[idx, :]})
        R = []
        for r in reward[0]:
            R.append(float(r))
        reward_real = train_reward[idx, :]
        plt.plot(R, "r")
        plt.plot(reward_real, "g")
        plt.savefig("train_time" + str(self.draw_time) + ".png")
        self.draw_time += 1
        plt.cla()

    # 定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)


class Reward_Net():
    def __init__(self, env, lr=0.0001, model_file=None):
        self.draw_time = 0
        self.n_features = env.observation_dim  # 输入特征维度
        self.learning_rate = lr  # 学习率
        self.loss_alpha = 0.95  # 过去的loss占比
        self.batch = BATCHSIZE_CRITIC  # 批大小
        self.iter = ITER_CRITIC  # 迭代总次数
        self.n_actions = env.action_dim  # 动作空间维度,特别现在是连续动作
        # 1.1 输入层  构建输入层
        self.obs_action = tf.placeholder(tf.float32, shape=[None, self.n_features + self.n_actions])
        # 1.2 隐含层   100个神经元 + relu
        self.c_f1 = tf.layers.dense(inputs=self.obs_action, units=100, activation=tf.nn.relu)
        # 1.3 输出层   1个神经元
        self.reward = tf.layers.dense(inputs=self.c_f1, units=1)
        # 定义critic网络的损失函数,输入为td目标
        self.reward_target = tf.placeholder(tf.float32, [None, 1])
        self.c_loss = tf.reduce_mean(tf.square(self.reward_target - self.reward))  # 使用(r-r)^2作为损失函数
        self.c_train_op = tf.train.AdamOptimizer(0.0002).minimize(self.c_loss)
        # tf工程
        self.sess = tf.Session()  # 图构建完毕之后就可以创建tf工程
        # 初始化图中的变量,类似于C++中的实例化
        self.sess.run(tf.global_variables_initializer())
        # 6.定义保存和恢复模型
        self.saver = tf.train.Saver()
        # 模型保留
        if model_file is not None:
            self.restore_model(model_file)

    # 模型训练
    def fit_critic(self, batch_obs_act, batch_reward, save_dir=None, epoch=10):
        flag = 0
        train_obs_act = batch_obs_act
        train_reward = batch_reward
        # Update the reward to the
        # 采样得到的数据量
        N = train_reward.shape[0]
        # 获得对应数据的标签
        train_indicies = np.arange(N)
        loss_line = []  # 损失记录
        num = 0  # 评估次数
        ls = 0  # 评估损失
        for i in range(self.iter):
            # 打乱顺序
            np.random.shuffle(train_indicies)
            # >>>>>>>>>>训练神经网络>>>>>>>>>>
            for j in range(int(math.ceil(N / self.batch))):
                # 得到训练样本的索引
                start_idx = j * self.batch % N
                idx = train_indicies[start_idx:start_idx + self.batch]
                # 对模型网络进行训练
                self.sess.run([self.c_train_op],feed_dict={self.obs_action: train_obs_act[idx, :], self.reward_target: train_reward[idx, :]})
                # 计算损失
                loss = self.sess.run([self.c_loss], feed_dict={self.obs_action: train_obs_act[idx, :], self.reward_target: train_reward[idx, :]})
                loss_line.append(loss)
                # 对当前学习情况进行评估
                if num == 0:
                    ls = loss[0]
                else:
                    ls = self.loss_alpha * ls + (1 - self.loss_alpha) * loss[0]
                num += 1
                if i > epoch:
                    flag = 1
                    break
            print("第%d次学习,奖励网络的误差为%f" % (i, ls))
            if flag == 1:
                break
            # <<<<<<<<<<训练神经网络<<<<<<<<<<
        # 保存模型
        if save_dir is not None:
            self.save_model(save_dir)
        # 绘制损失变化图
        print("==============DRAW the picture============")
        # 对比计算情况
        reward = self.sess.run([self.reward], feed_dict={self.obs_action: train_obs_act[idx, :]})
        R = []
        for r in reward[0]:
            R.append(float(r))
        reward_real = train_reward[idx, :]
        plt.plot(R, "r")
        plt.plot(reward_real, "g")
        plt.savefig("train_time" + str(self.draw_time) + ".png")
        self.draw_time += 1
        plt.cla()
    # 定义存储模型函数

    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)