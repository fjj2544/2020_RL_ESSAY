import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from env.plane.liner_env import  Planes_Env
import tensorflow as tf
import numpy as np
import math
import gym
import matplotlib.pyplot as plt
RENDER = False
import random


#这个很关键也很简单,类似于一个采样器
class Sample():
    def __init__(self,env, policy_net):
        self.env = env # 环境模型
        self.policy_net=policy_net # 策略网络
        self.gamma = 0.90 # 折扣累计回报
    '''
    随机策略生成数据
    根据环境和策略采集N条轨迹
    s_t a_t r_t s_(t+1)
    '''
    def sample_normalize(self, num_episodes):
        batch_obs_next = []
        batch_obs=[]
        batch_actions=[]
        batch_r =[]
        for i in range(num_episodes):
            observation = self.env.reset()
            #将一个episode的回报存储起来
            # reward_episode = []
            while True:
                # if RENDER:self.env.render()
                #根据策略网络产生一个动作
                action = [np.random.uniform(env.action_bound[0],env.action_bound[1])] # action between [-2,2)
                observation_, reward, done, info = self.env.step(action)
                #存储当前观测
                batch_obs.append(np.reshape(observation,[1,3])[0,:])
                #存储后继观测
                batch_obs_next.append(np.reshape(observation_,[1,3])[0,:])
                # print('observation', np.reshape(observation,[1,3])[0,:])
                #存储当前动作
                batch_actions.append(action)
                #存储轨迹回报目前还没有用到回报这个东西
                batch_r.append((reward+8)/8)
                # reward_episode.append((reward+8)/8)
                #一个episode结束
                if done:
                    #处理数据
                    # print(self.delta)
                    break
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.env.observation_dim]) # 时间标签 * 状态数据 ,时间标签作为索引不作为数据
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs_next), self.env.observation_dim]) # 时间标签 * 状态数据
        batch_actions = np.reshape(batch_actions,[len(batch_actions),1]) #时间标签 动作,就是一个一维向量
        delta = batch_obs_next - batch_obs  # 状态的差值,学习一个状态的变化量 学习 f()*dt 积分项
        self.obs_mean, self.obs_std = self.normalize(batch_obs) # 求取一个batch的均值和方差
        self.delta_mean, self.delta_std = self.normalize(delta)
        self.action_mean, self.action_std = self.normalize(batch_actions)
        self.obs_action_mean = np.hstack((self.obs_mean, self.action_mean)) # 这个要求行相等 ,当前的行是时间标签,或者说第一维度为时间标签后面为数据
        self.obs_action_std = np.hstack((self.obs_std, self.action_std))  # 和上面类似
        return self.obs_action_mean, self.obs_action_std, self.delta_mean,self.delta_std
    '''
    策略网络生成数据
    '''
    # 产生num_episodes条轨迹,这个才是真正的采集num_episodes 条轨迹
    def sample_episodes(self, num_episodes):
        batch_obs_next = []
        batch_obs=[]
        batch_actions=[]
        batch_r =[]
        for i in range(num_episodes):
            observation = self.env.reset()
            #将一个episode的回报存储起来
            # reward_episode = []
            while True:
                #根据策略网络产生一个动作
                state = np.reshape(observation,[1,3])
                action = self.policy_net.choose_action(state)
                observation_, reward, done, info = self.env.step(action)
                #存储当前观测
                batch_obs.append(np.reshape(observation,[1,3])[0,:])
                #存储后继观测
                batch_obs_next.append(np.reshape(observation_,[1,3])[0,:])
                # print('observation', np.reshape(observation,[1,3])[0,:])
                #存储当前动作
                batch_actions.append(action)
                #存储立即回报
                batch_r.append((reward+8)/8)
                # reward_episode.append((reward+8)/8)
                #一个episode结束
                if done:
                    break
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.env.observation_dim])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs), self.env.observation_dim])
        batch_delta = batch_obs_next - batch_obs
        batch_actions = np.reshape(batch_actions,[len(batch_actions),1])
        batch_rs = np.reshape(batch_r,[len(batch_r),1])
        batch_obs_action= np.hstack((batch_obs,batch_actions))
        return batch_obs_action, batch_delta,batch_obs_next
    def normalize(self, batch_data):
        mean = np.mean(batch_data,0) # 难道仅仅学习一个均值 axis =0 对每一列求均值
        std = np.std(batch_data,0) #难道每一幕仅仅学习一个均值
        return mean, std
# 类似于策略参数化了 P(mu,std)
# 可清空缓冲区,防止干扰
## 也就是这个地方需要改成PI2的形式
## 我现在需要策略网络采样,而且到了非线性我可能不能用所谓的随机采样的方法解决问题,当然有约束的随机采样倒是可以
class Policy_Net():
    def __init__(self, env, action_bound, lr = 0.0001, model_file=None):
        tf.reset_default_graph()
        self.learning_rate = lr
        #输入特征的维数,shape[0] 应该就是有多少行也就是索引维度是多少
        self.n_features = env.observation_dim
        #输出动作空间的维数
        self.n_actions = 1
        #1.1 输入层
        self.obs = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.pi, self.pi_params = self.build_a_net('pi', trainable=True)
        self.oldpi, self.oldpi_params = self.build_a_net('oldpi', trainable=False)
        # 保证action在可行的输入以内
        print("action_bound",action_bound[0],action_bound[1])
        self.action = tf.clip_by_value(tf.squeeze(self.pi.sample(1),axis=0), action_bound[0], action_bound[1])
        #定义新旧参数的替换操作
        self.update_oldpi_op = [oldp.assign(p) for p,oldp in zip(self.pi_params, self.oldpi_params)]
        #1.5 当前动作，输入为当前动作，delta,
        self.current_act = tf.placeholder(tf.float32, [None,1])
        #优势函数
        self.adv = tf.placeholder(tf.float32, [None,1])
        #2. 构建损失函数
        ratio = self.pi.prob(self.current_act)/self.oldpi.prob(self.current_act)
        #替代函数
        surr = ratio*self.adv
        self.a_loss = -tf.reduce_mean(tf.minimum(surr,tf.clip_by_value(ratio, 1.0-0.2, 1.0+0.2)*self.adv))
        # self.loss += 0.01*self.normal_dist.entropy()
        #3. 定义一个动作优化器
        self.a_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss)
        #4.定义critic网络
        self.c_f1 = tf.layers.dense(inputs=self.obs, units=100, activation=tf.nn.relu)
        self.v = tf.layers.dense(inputs=self.c_f1, units=1)
        #定义critic网络的损失函数,输入为td目标
        self.td_target = tf.placeholder(tf.float32, [None,1])
        self.c_loss = tf.reduce_mean(tf.square(self.td_target-self.v))
        self.c_train_op = tf.train.AdamOptimizer(0.0002).minimize(self.c_loss)
        # 5. tf工程
        self.sess = tf.Session()
        #6. 初始化图中的变量
        self.sess.run(tf.global_variables_initializer())
        #7.定义保存和恢复模型
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
    def build_a_net(self, name, trainable):
        with tf.variable_scope(name):
            # 1.2.策略网络第一层隐含层
            self.a_f1 = tf.layers.dense(inputs=self.obs, units=100, activation=tf.nn.relu,trainable=trainable)
            # 1.3 第二层，均值
            a_mu = 2*tf.layers.dense(inputs=self.a_f1, units=self.n_actions, activation=tf.nn.tanh,trainable=trainable)
            # 1.3 第二层，标准差
            a_sigma = tf.layers.dense(inputs=self.a_f1, units=self.n_actions, activation=tf.nn.softplus,trainable=trainable)
            # a_mu = 2 * a_mu
            a_sigma = a_sigma
            # 定义带参数的正态分布
            normal_dist = tf.contrib.distributions.Normal(a_mu, a_sigma)
            # 根据正态分布采样一个动作
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return normal_dist, params
    def get_v(self, state):
        v = self.sess.run(self.v, {self.obs:state})
        return v
    #依概率选择动作
    def choose_action(self, state):
        action = self.sess.run(self.action, {self.obs:state})
        # print("greedy action",action)
        return action[0]
    #定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    #定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
## 很简单的动力学网络,包含一个采样器
## 但是对于env必须是统一定义,不能是随意定义,当然理解之后就可以考虑自己改变,但是最好还是改env
## s_(t+1) = s_(t) + f(s_t,a_t) * dt : dynamic_net = f(s_t,a_t)
'''
这里相当于实例化了一个神经网络
input =  tf.placeholder(数据类型,shape=[None,输入维度]) 比如说3个状态2个动作就是 3+2,当然动作是连续的,状态指的是马赫，俯仰角..
hidden1 = tf.layers.dense(inputs=input,units= 神经元数量) 
hidden2 = tf.layers.dense(inputs=hidden1,units= 神经元数量) 
output = tf.layers.dense(inputs = hidden2,units=  输出维度)
self.delta =  tf.placeholder(tf.float32,[None, self.n_features]) #这个代表实际的输出(正确标签)
self.loss = tf.reduce_mean(tf.square(self.predict-self.delta)) # 居然是降低维度的,计算每一个元素的平方
# 2. 构建损失函数
self.delta =  tf.placeholder(tf.float32,[None, self.n_features]) #说白了tf给出一个一种计算的流程或者说计算的方法
self.loss = tf.reduce_mean(tf.square(self.predict-self.delta))
# 3. 定义一个优化器
self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
# 4. tf工程
self.sess = tf.Session() # 图构建完毕之后就可以创建tf工程
# 5. 初始化图中的变量,类似于C++中的实例化
self.sess.run(tf.global_variables_initializer())
# 6.定义保存和恢复模型,内部存储,并没有存储成文件形式
self.saver = tf.train.Saver()
# 迭代训练
if model_file is not None:
    self.restore_model(model_file)
'''
class Dynamic_Net():
    def __init__(self,env, sampler, lr=0.0001, model_file=None):
        self.n_features = env.observation_dim # 输入特征维度
        self.learning_rate = lr # 学习率
        self.loss_alpha = 0.95 # 过去的loss占比
        self.sampler = sampler # 采样器
        self.sampler.sample_normalize(200) # 样本正则化
        '''估计采样器的均值和方差'''
        self.obs_action_mean = self.sampler.obs_action_mean # 动作均值
        self.obs_action_std = self.sampler.obs_action_std # 动作方差
        self.delta_mean = self.sampler.delta_mean # 改变量均值
        self.delta_std = self.sampler.delta_std # 改变量方差
        # 得到数据的均值和协方差,产生100条轨迹
        self.batch = 50 # 批大小
        self.iter = 2000 # 迭代总次数
        self.n_actions = 1 # 动作空间维度,特别现在是连续动作
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
    def fit_dynamic(self):
        flag = 0
        batch_obs_act, batch_delta,_ = self.sampler.sample_episodes(100)
        # 0,1标准化
        # train_obs_act = (batch_obs_act-self.obs_action_mean)/(self.obs_action_std)
        # train_delta = (batch_delta-self.delta_mean)/(self.delta_std)
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
                if ls < 1e-10 or i > 100:
                    flag=1
                    break
            print("第%d次实验,误差为%f" % (i, ls))
            if flag == 1:
                break
        #保存模型
        self.save_model('./current_best_dynamic_fit_pendulum')
        #显示训练曲线
        number_line=np.arange(len(loss_line)) # 获得训练次数的标签
        plt.plot(number_line, loss_line)
        plt.show()
    def prediction(self,s_a, target_state):
        #正则化数据
        norm_s_a = s_a
        #利用神经网络进行预测
        delta = self.sess.run(self.predict, feed_dict={self.obs_action:norm_s_a})
        predict_out = delta + s_a[:,0:3]
        x = np.arange(len(predict_out))
        plt.figure(1)
        plt.plot(x, predict_out[:,0],)
        plt.plot(x, target_state[:,0],'--')
        plt.savefig("./alpha.png")
        # plt.figure(11)
        # plt.plot(x, s_a[:,0])
        # plt.plot(x,predict_out[:,0],'--')
        # plt.plot(x, target_state[:,0],'-.')
        plt.figure(2)
        plt.plot(x, predict_out[:, 1])
        plt.plot(x, target_state[:, 1],'--')
        plt.savefig("./theta.png")

        plt.figure(3)
        plt.plot(x, predict_out[:, 2])
        plt.plot(x, target_state[:, 2],'--')
        plt.savefig("./q.png")

        plt.show()
        return predict_out
    # 定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)
    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)

if __name__=='__main__':
    '''创建仿真环境'''
    env = Planes_Env()
    env.seed(1)
    print("state dim:",env.observation_dim,"action dim :",env.action_dim)
    '''初始化'''
    # 创建动作的范围
    action_bound = env.action_bound
    # 实例化策略网络,采用策略网络
    brain = Policy_Net(env, action_bound)
    # 实例化采样器,这个数据结构很关键
    sampler = Sample(env, brain)
    '''随机采集200条轨迹,然后得到轨迹的均值和标准差 相当于一个batch'''
    # print(sampler.sample_normalize(200)) # 测试采集200条轨迹
    # print(sampler.obs_action_mean, sampler.obs_action_std,sampler.delta_mean,sampler.delta_std) # 测试采样器方差和均值

    # ob_a,ob_delta,ob_next= sampler.sample_episodes(1) # 测试采集一条轨迹数据
    # 一条轨迹的长度
    # print(ob_a.shape,ob_delta.shape,ob_next.shape) # 便于编程 input output  // next_input = output + input 有没有点递归神经网络的感觉
    # 初始化动力学网络-----model_based

    # '''拟合动力学模型'''
    # dynamic_net = Dynamic_Net(env, sampler)
    # dynamic_net.fit_dynamic()

    '''测试拟合动力学模型的效果'''
    dynamic_net = Dynamic_Net(env, sampler,model_file='./tf_model/current_best_dynamic_fit_pendulum')
    batch_obs_act, batch_delta, target_state = sampler.sample_episodes(1)
    predict_obs = dynamic_net.prediction(batch_obs_act,target_state)
    print(np.fabs(predict_obs[:,2]-target_state[:,2]))
