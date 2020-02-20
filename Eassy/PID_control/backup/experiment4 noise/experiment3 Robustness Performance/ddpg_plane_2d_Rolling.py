"""
这里我初步尝试滚动优化
1. 首先我会设计一个接口,可以模拟K步得到当前env
2. 然后我会设计PI2的优化
3. 封装好前面的内容
"""

import pygame
import numpy as np
import tensorflow as tf
# from load import *
from pygame.locals import *
import math
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import multiprocessing as mp
import numba
from numba import jit
import copy



Init_K = [1000,1000,1000]
RENDER = False
C_UPDATE_STEPS = 10
A_UPDATE_STEPS = 1

linewidth = 1
fontsize = 5
markersize = 1


"""
空气动力学模型
"""


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

"""
空气动力学模型
"""

class Planes_Env:
    def __init__(self):
        # self.actions = [0,1]
        self.observation_dim = 2
        # 状态量分别为[飞机迎角 alpha, 飞机俯仰角theta, 飞机俯仰角速度q]
        self.state = np.array([0.0, 0.0, 0.0])
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        self.max_steps = 200
        self.viewer = None
        # 飞机
        self.b_alpha = 0.073
        self.b_delta_z = -0.0035
        self.a_alpha = 0.7346
        self.a_delta_z = -2.8375
        self.a_q = 3.9779
        self.delta_z_mag = 0.1
        self.tau = 0.02
        self.theta_desired = 10
        # 角度阈值 攻角alpha[-1,10];舵偏delta_z[-20,20]
        self.alpha_threshold_max = 10
        self.alpha_threshold_min = -1
        self.delta_z_threhold_max = 20
        self.delta_z_threhold_min = -20
        self.reward = 0.0
        self.cost = 0.0

        self.delta_z = 0.0
        # 最大加速度,dez最大变化量
        self.max_delta_z_change = 1
        self.delta_z_change = 0
        self.last_delta_z = 0.0

    def reset(self):
        n = np.random.randint(1, 1000, 1)
        np.random.seed(n)
        self.state = np.zeros((3, 1), dtype=np.float64)
        # self.state = np.random.uniform(-0.5, 0.5, size=(3,))
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        return self.observation

    def step(self, action):
        action = action[0]
        state = self.state
        alpha, theta, q = state
        observation_pre = theta - self.theta_desired
        ## 非线性约束
        self.delta_z_change = np.clip(action,self.delta_z_threhold_min,self.delta_z_threhold_max) - self.last_delta_z
        self.delta_z = np.clip(self.delta_z_change +np.random.normal(0,1) ,-self.max_delta_z_change,self.max_delta_z_change) +self.last_delta_z
        self.last_delta_z = self.delta_z

        # 动力学方程 攻角alpha，俯仰角theta 俯仰角速度q  舵偏delta_z
        alpha_dot = q - self.b_alpha * alpha - self.b_delta_z * self.delta_z
        theta_dot = q
        q_dot = -self.a_alpha * alpha - self.a_q * q - self.a_delta_z * self.delta_z
        # 积分得到状态量
        q = q + self.tau * q_dot+np.random.normal(0,1)


        theta = theta + self.tau * theta_dot
        observation_cur = theta - self.theta_desired
        alpha = np.clip(alpha + self.tau * alpha_dot+np.random.normal(0,1)  ,self.alpha_threshold_min,self.alpha_threshold_max)


        self.steps_beyond_done += 1
        self.state = np.array([alpha, theta, q])
        # 根据更新的状态判断是否结束
        lose = alpha < self.alpha_threshold_min or alpha > self.alpha_threshold_max
        # TODO:设置回报
        if not lose:
            self.reward = -((theta - self.theta_desired) ** 2 + 0.1 * q ** 2 + 0.01 * action ** 2)
        else:
            self.reward = -2500
        done = lose or self.steps_beyond_done > self.max_steps
        self.observation = np.array([observation_pre, observation_cur])
        return self.observation, self.reward, done


''' 要有行之有效的方法,能够解决一类自动整定问题 随机设置一个初始点都能收敛到一个较好的点  '''
## 目标量,采用动态目标
Overshoot_target = 1e-3
ts_target = 150
Waveform_oscillation_bound = 1
Static_error_bound = 0.01

# TODO:计算调整时间
## 调整时间的计算范围,连续K次测试,保证K次测试内能通过
adjust_bound = 0.02
#可信度 用于计算调整时间
belief_times = 50

class PID_model():
    def __init__(self):
        self.env = Planes_Env()

    def get_epsolid_reward(self,env, k1=1.5, k2=2.5, k3=0.5,is_test = False):
        total_step = 2000
        self.env = copy.copy(env)
        # self.env.state = env.state
        # self.env.observation = env.observation
        # self.env.steps_beyond_done = env.steps_beyond_done
        alpha = []
        theta = []
        desired_theta = []
        q = []
        time = []
        i = 1
        control = []
        ierror = 0
        derror_list = []
        error_list = []
        dez_list = []
        # 峰值时间
        tp = 0
        ''' 计算调整时间   如果调整时间过大我们就加大惩罚 '''
        ts = total_step
        count = 0
        for i in range(total_step):
            if count >= belief_times and ts == total_step:
                ts = i
                break
            error = self.env.theta_desired - self.env.state[1]
            derror = -self.env.state[2]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror

            dez_list.append(action)
            if (error == 0 and tp == 0):
                tp = i
            self.env.step(np.array([action]))
            alpha.append(self.env.state[0])
            theta.append(self.env.state[1])
            desired_theta.append(self.env.theta_desired)
            q.append(self.env.state[2])
            time.append(i)
            control.append(action)

            if(abs(error)<=abs(adjust_bound * self.env.theta_desired)):
                count += 1
            else:
                count = 0
            # ## 分阶段优化，因为每个阶段的任务应该是不同的,模拟人的思想，模拟我们自己的调参经验,先得到一个可行解，然后转移得到带有约束的最优解

            ## 虽然我觉得这里应该加入极大值限制,这里是不是应该改环境
            if self.env.state[0] < self.env.alpha_threshold_min or self.env.state[0] > self.env.alpha_threshold_max:
                count += 1

        # 超调量 kp
        Overshoot = max(abs(np.array(theta))) - max(abs(np.array(desired_theta)))
        Overshoot = 0 if Overshoot < Overshoot_target else (Overshoot - Overshoot_target) / Overshoot_target
        # 调整时间
        ts = 0 if ts <= ts_target else (ts - ts_target) / ts_target
        r = Overshoot + ts
        return r

    def get_new_env(self, env,step_time ,k1=1.5, k2=2.5, k3=0.5):


        self.env = copy.copy(env)
        # print("before we come ",self.env.state[1],env.state[1])
        alpha = []
        theta = []
        desired_theta = []
        q = []
        time = []
        i = 0
        control = []
        ierror = 0
        derror_list = []
        error_list = []
        action_list = []
        dez_list = []
        while True:
            if i == step_time:
                break

            error = self.env.theta_desired - self.env.state[1]
            derror = -self.env.state[2]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror

            action_list.append(action)
            dez_list.append(self.env.delta_z)

            self.env.step(np.array([action]))
            alpha.append(self.env.state[0])
            theta.append(self.env.state[1])
            desired_theta.append(self.env.theta_desired)
            q.append(self.env.state[2])
            time.append(i)
            control.append(action)
            i = i + 1
        # print("after we come ", self.env.state[1], env.state[1])
        return self.env,alpha, dez_list, theta, desired_theta
    def model_simulation( self,k1=1.5, k2=2.5, k3=0.5, total_step=200):
        self.env.reset()
        alpha = []
        theta = []
        desired_theta = []
        q = []
        time = []
        i = 1
        control = []
        ierror = 0
        derror_list = []
        error_list = []
        action_list = []
        dez_list = []


        while i < total_step:
            """ FOR DEBUG """
            # if i % 10 == 0:
            #     print(i,self.env.state[1],self.env.theta_desired)
            error = self.env.theta_desired - self.env.state[1]
            derror = -self.env.state[2]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror

            action_list.append(action)
            dez_list.append(self.env.delta_z)


            self.env.step(np.array([action]))
            alpha.append(self.env.state[0])
            theta.append(self.env.state[1])
            desired_theta.append(self.env.theta_desired)
            q.append(self.env.state[2])
            time.append(i)
            control.append(action)

            i = i + 1
        return alpha, dez_list, theta, desired_theta


""" -----------------------------------------------------------随机初始化参数-------------------------------------------------------------"""
Ki_Min = 0
Ki_Max = 100.0
Kp_Min = 0
Kp_Max = 100.0
Kd_Min = 0
Kd_Max = 100.0

""" -----------------------------------------------------------归一化部分-------------------------------------------------------------"""
"""Z-score normaliaztion"""
"""这种方法要求原始数据的分布可以近似为高斯分布，否则效果会很差。标准化公式如下 """
def ZscoreNormalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
"""[0,1] normaliaztion"""
def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


""" -----------------------------------------------------------强化学习部分-------------------------------------------------------------"""
# 训练次数,也就是策略迭代次数
training_times = 100  # training times
# 路径数量,也就是策略评估数量
roll_outs = 20  # path number
# 后期可能加入延迟更新
Max_Adjust_times = 1000
class RL_PI2:
    def __init__(self,if_filter=True,attenuation_step_length=10,alpha=0.85):
        ## 滚动优化,用于记录当前阶段
        self.env = Planes_Env()

        # 记录每次策略迭代之后的K(包括初始化）   动态变量 每一次策略迭代刷新一次
        self.K = np.zeros((3, 1), dtype=np.float64)
        # 记录每 roll_outs 局势内的 K  动态变量 每一次策略迭代刷新一次
        self.K_roll = np.zeros((3, roll_outs), dtype=np.float64)
        # 记录策略迭代中全部的K     静态变量 每次运行才会刷新一次
        self.K_record = np.zeros((3, roll_outs, training_times), dtype=np.float64)
        # 噪声方差,利用不同的方差控制不同参数的调节率, 从而实现分阶段控制         动态变量,会逐步衰减   是否需要衰减还需我们再度考虑
        self.sigma = np.zeros((3, 1), dtype=np.float64)
        # K变化量,用于存储噪声大小,进行策略改进        动态变量 每一次策略迭代刷新一次
        self.k_delta = np.zeros((3, roll_outs), dtype=np.float64)
        # 记录回报,用于策略改进   动态变量 每一次策略迭代刷新一次
        self.loss = np.zeros((roll_outs, 1), dtype=np.float64)
        # 记录所有策略迭代中损失函数  静态变量 每次运行才会刷新一次
        self.loss_record = np.zeros((roll_outs, training_times), dtype=np.float64)
        # 每次策略迭代之后的损失函数    动态变量 每一次策略迭代刷新一次
        self.loss_after_training = np.zeros((training_times+Max_Adjust_times, 1), dtype=np.float64)
        # 每次策略迭代之后的K  动态变量 每一次策略迭代刷新一次
        self.K_after_training = np.zeros((3, training_times+Max_Adjust_times), dtype=np.float64)

        # 每次滚动优化的K
        self.K_after_roll_step = np.zeros((3,2000),dtype=np.float64)
        """ -----------------------------------------------------------定义算法超参数-------------------------------------------------------------"""
        # 衰减频率
        self.attenuation_step_length = attenuation_step_length
        # 衰减系数 这里我觉得可以加入先验知识,不等权衰减
        self.alpha = alpha
        # 记录当前是第几幕
        self.current_roll = 0
        # 记录当前第几次策略迭代
        self.current_training = 0
        # PI2超参数
        self.PI2_coefficient = 30.0
        ## 交互模型
        self.reward_model = PID_model()
        ## 是否记录数据
        self.save_data = False
        ## 是否随机初始化
        self.random_init = False
        ## 是否绘图
        self.plot_photo = True
        ## 是否记录图片
        self.save_photo = True
        ## 是否filter
        self.if_filter = if_filter

        ##记录更新的初值
        self.K0 = np.zeros((3, 1), dtype=np.float64)
        ## 记录每次优化时间
        self.FR_time = np.zeros((1000,1), dtype=np.float64)
    def data_record(self):
        np.savetxt('./data/loss_after_training.txt',self.loss_after_training)
        np.savetxt('./data/K_after_training.txt',self.K_after_training)

    # 理论上来看学习优化降低了方差
    def sample_using_PI2(self):  # using PI2 to sample
        time_start = time.time()
        print('start')
        # 初始化参数
        self.set_initial_value()
        # 开始训练
        # self.training()
        self.rolling_optimization()
        time_end = time.time()
        print('end')
        print('total time', time_end - time_start)

        if self.save_data:
            self.data_record()

    def set_initial_value(self,INIT_K=[1.5,2.5,0.5]):

        if self.random_init:
            self.K[0] = random.uniform(Kp_Min,Kp_Max)
            self.K[1] = random.uniform(Kd_Min,Kd_Max)
            self.K[2] = random.uniform(Ki_Min,Ki_Max)
        else:
            self.K0 = INIT_K
            self.K[0] = INIT_K[0]
            self.K[1] = INIT_K[1]
            self.K[2] = INIT_K[2]
        # 初始化方差
        self.sigma[0] = 1.0
        self.sigma[1] = 0.3
        self.sigma[2] = 0.1
        # 初始化记录参数
        self.current_roll = 0
        self.current_training = 0
        # 初始化滚动环境
        self.env.reset()

    """ -----------------------------------------------------------计算轨迹回报,用于并行------------------------------------------------------------"""
    @jit(forceobj=True,nopython=True,nogil=True)
    def cal_trajectory_loss(self, j):
        self.current_roll = j
        delta1 = np.random.normal(0, self.sigma[0], 1)
        delta2 = np.random.normal(0, self.sigma[1], 1)
        delta3 = np.random.normal(0, self.sigma[2], 1)
        cur_k1 = self.K[0] + delta1
        cur_k2 = self.K[1] + delta2
        cur_k3 = self.K[2] + delta3
        loss = self.reward_model.get_epsolid_reward(self.env,cur_k1, cur_k2, cur_k3)
        # ##如果没有优势,那么我们就可以不去学习,没有必要浪费时间去学习没有用的东西
        if self.if_filter and (self.current_training > 1 and loss > self.loss_after_training[self.current_training - 1]):
            delta1 = delta2 = delta3 = 0.0
            cur_k1 = self.K[0] + delta1
            cur_k2 = self.K[1] + delta2
            cur_k3 = self.K[2] + delta3
        return delta1,delta2,delta3,cur_k1,cur_k2,cur_k3,loss
    """ -----------------------------------------------------------策略评估------------------------------------------------------------"""
    @jit(forceobj=True, nopython=True, nogil=True,parallel=True)
    def policy_evl(self):
        # 采样N局,记录稀疏回报
        multi_res = [poll.apply_async(self.cal_trajectory_loss, (j,)) for j in range(roll_outs)]
        for j, res in enumerate(multi_res):
            self.k_delta[0, j] = res.get()[0]
            self.k_delta[1, j] = res.get()[1]
            self.k_delta[2, j] = res.get()[2]
            self.K_roll[0, j] = res.get()[3]
            self.K_roll[1, j] = res.get()[4]
            self.K_roll[2, j] = res.get()[5]
            self.loss[j] = res.get()[6]
            self.loss[j] = self.loss[j] + np.random.uniform(-0.02, 0.02, 1)

    """ -----------------------------------------------------------策略改善------------------------------------------------------------"""
    def policy_improve(self):
        exponential_value_loss = np.zeros((roll_outs, 1), dtype=np.float64)  #
        probability_weighting = np.zeros((roll_outs, 1), dtype=np.float64)  # probability weighting of each roll
        # 果然这里要做一个max min 标准化
        for i2 in range(roll_outs):
            exponential_value_loss[i2] = np.exp(-self.PI2_coefficient * (self.loss[i2] - self.loss.min())
                                                / (self.loss.max() - self.loss.min()))
        for i2 in range(roll_outs):
            probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)

        temp_k = np.dot(self.k_delta, probability_weighting)

        self.K = self.K + temp_k
    def iterator_finished(self):
        flag1 = sum((self.K_after_training[:, self.current_training - 1] - self.K_after_training[:,
                                                                   self.current_training]) ** 2) <= 1e-6
        flag2 = self.loss_after_training[self.current_training]
        if flag1 < 1e-6 and flag2 < 0.5:
            return True
        else :
            return False
    """ ----------------------------------------------------------策略迭代部分------------------------------------------------------------"""
    def training(self):
        i =0
        self.K_after_training[:, self.current_training] = self.K[:, 0]
        self.loss_after_training[self.current_training] = self.reward_model.get_epsolid_reward(self.env, self.K[0],
                                                                                               self.K[1],self.K[2])
        while i < training_times:
            # 分阶段优化 首先调整到局部最优 然后找到带有约束的满意解
            i += 1
            self.current_training = i
            # 方差衰减和可视化
            if self.current_training % self.attenuation_step_length == 0  and self.current_training != 0:
                self.sigma = self.sigma / self.alpha  # attenuation
                if self.current_training %10 == 0:
                    plt.plot(self.loss_after_training[self.current_training - 10:self.current_training])
                    plt.title("loss between %d and %d epoch"%(self.current_training - 10,self.current_training))
                    plt.show()
            # 策略迭代框架
            self.policy_evl()
            self.policy_improve()
            # 记录参数
            self.K_after_training[:, self.current_training] = self.K[:, 0]

            self.loss_after_training[self.current_training] = self.reward_model.get_epsolid_reward(self.env,self.K[0], self.K[1],
                                                                           self.K[2])
            if self.iterator_finished():
                break
            """ ----------------------------------------------------------虽然可以用但是这一块有BUG------------------------------------------------------------"""
            # 输出当前训练次数
            if(self.current_training % self.attenuation_step_length == 0 ):
                print(self.current_training,time.time()-first_time)
        ## 总共有training_times+1 条数据
        for it in range(self.current_training,training_times+1):
            self.loss_after_training[it] = self.loss_after_training[self.current_training]
        return self.K[0],self.K[1],self.K[2],self.K_after_training,self.loss_after_training,self.current_training+1
    def rolling_optimization(self,rolling_time=20,total_step=200):
        opt_times = int(total_step/rolling_time)
        theta_list = []
        theta_desired_list =[]
        alpha_list =[]
        delta_z_list = []
        K_list  =[]
        loss_list = []
        iterator = 0
        self.K_after_roll_step[:, iterator] = self.K[:, 0]
        max_time = 0
        while iterator < opt_times :
            # 每次策略迭代之后的损失函数    动态变量 每一次策略迭代刷新一次
            self.loss_after_training = np.zeros((training_times + Max_Adjust_times, 1), dtype=np.float64)
            # 每次策略迭代之后的K  动态变量 每一次策略迭代刷新一次
            self.K_after_training = np.zeros((3, training_times + Max_Adjust_times), dtype=np.float64)
            # print("test env 1",self.env.state[1]) 这个很关键
            self.current_training = 0
            self.training()
            max_time = max_time+ self.current_training
            ## 迭代多少次和当前训练没有关系
            self.FR_time[iterator] = self.current_training
            for i in range(self.current_training+1):
                K_list.append(self.K_after_training[:,i])
                loss_list.append(self.loss_after_training[i])
            plt.plot(self.loss_after_training)
            plt.title("FUCK11111111")
            # print("test env 2 ",self.env.state[1]) alpha, dez_list, theta, desired_theta
            iterator += 1
            cur_step = iterator * rolling_time
            self.K_after_roll_step[:,iterator]=self.K[:,0]
            self.env,alpha, delta_z, theta, theta_desired = self.reward_model.get_new_env(self.env,rolling_time,self.K[0],self.K[1],self.K[2])
            for item in alpha:
                alpha_list.append(item)
            for item in delta_z:
                delta_z_list.append(item)
            for item in theta:
                theta_list.append(item)
            for(item) in theta_desired:
                theta_desired_list.append(item)
            if iterator % 2==0:
                plt.figure(2)
                plt.plot(theta_list, 'b+', label="time-theta")
                plt.plot(theta_desired_list, 'r', label="time-desired_theta")
                plt.legend(loc="best")
                plt.title("Rolling optimization graph After %d Opt" % iterator)
                plt.show()
        plt.figure(2)
        plt.plot(theta_list, 'b+', label="time-theta")
        plt.plot(theta_desired_list, 'r', label="time-desired_theta")
        plt.legend(loc="best")
        plt.title("Rolling optimization graph After %d Opt" % iterator)
        plt.show()
        label = ["Kp", "Kd", "Ki"]
        color = ["r", "g", "b", "k"]
        line_style = ["-", "--", ":", "-."]
        marker = ['*', '^', 'h']
        "绘制K曲线"
        plt.xticks(fontproperties='Times New Roman')
        plt.yticks(fontproperties='Times New Roman')
        plt.xlabel("Number of Rolling Optimization")
        plt.ylabel("$\mathcal{K}^{*}$ Value")
        for i in range(3):
            plt.plot(self.K_after_roll_step[i][:iterator+1], label=label[i], color=color[i],
                     linestyle=line_style[i], marker=marker[i])
        plt.legend(loc='best', prop={'family': 'Times New Roman'})
        # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混
        plt.title("$\mathcal{K}^{*}$ Iteration Graph", fontdict={'family': 'Times New Roman'})
        save_figure("./photo/exp1/", "K_Rolling_interval_%d_%d_%d_%d.pdf"%(rolling_time,self.K0[0],self.K0[1],self.K0[2]))
        plt.show()
        plt.figure(figsize=(2.8, 1.7), dpi=300)

        plt.xticks(fontproperties='Times New Roman',fontsize=fontsize)
        plt.yticks(fontproperties='Times New Roman',fontsize=fontsize)
        plt.xlabel("Number of Rolling Optimization",fontproperties='Times New Roman', fontsize=fontsize)
        plt.ylabel("Number of iterations",fontproperties='Times New Roman', fontsize=fontsize)
        "设置坐标轴"
        x_major_locator = plt.MultipleLocator(1)
        # 把x轴的刻度间隔设置为1，并存在变量里
        y_major_locator = plt.MultipleLocator(1)
        # 把y轴的刻度间隔设置为10，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        # 把y轴的主刻度设置为10的倍数
        plt.ylim(0, 12)
        # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
        plt.plot(self.FR_time[:iterator], label='Number of iterations', color=color[0],
                 linestyle=line_style[0], marker='^', markersize=4.5, linewidth=linewidth)
        plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': fontsize})
        # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混
        # plt.title("$\mathcal{K}^{*}$ Iteration Graph",fontdict={'family': 'Times New Roman'})
        save_figure("./photo/exp1/",
                    "time.pdf" )
        plt.show()

        return alpha_list,delta_z_list,theta_list,theta_desired_list,K_list,loss_list,max_time+1

def save_data(dir,name,data):
    mkdir(dir)
    np.savetxt(dir+name,data)
def read_data(dir):
    data = np.loadtxt(dir)
    return data
def save_figure(dir,name):
    mkdir(dir)
    plt.savefig(dir+name,bbox_inches = 'tight')

def plot_result(alpha_list,delta_z_list,theta_list,theta_desire_list,figure_number = 3):
    label = ["FR-PI2", "F-PI2", "PI2", "Reference parameter"]
    color = ["r", "g", "b", "k"]
    line_style = ["-", "--", ":", "-."]
    marker = ['*', '^', 'h']

    "绘制$\\alpha$曲线"
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xlabel("Time$(0.02s)$")

    plt.ylabel("Attack Angle $(Degree)$")
    for i in range(figure_number):
        plt.plot(alpha_list[i], label=label[i],color=color[i],linestyle=line_style[i])

    plt.legend(loc='best', prop={'family': 'Times New Roman'})
    # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混
    # plt.title("$\\alpha$ Control Curve ", fontdict={'family': 'Times New Roman'})
    save_figure("./photo/exp1/", "alpha_Curve.pdf")
    plt.show()

    "绘制delta_z曲线"

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xlabel("Time$(0.02s)$")

    plt.ylabel("Pitch Rudder Angle $(Degree)$")
    for i in range(figure_number):
        plt.plot(delta_z_list[i], label=label[i], color=color[i], linestyle=line_style[i])

    # plt.title("$\\delta_z$  Control Curve ", fontdict={'family': 'Times New Roman'})
    plt.legend(loc='best', prop={'family': 'Times New Roman'})
    save_figure("./photo/exp1/", "delta_z_Curve.pdf")
    plt.show()
    "绘制theta曲线"
    plt.figure(num=2)

    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xlabel("Time$(0.02s)$")

    plt.ylabel("Pitch Angle $(Degree)$")

    for i in range(figure_number):
        plt.plot(theta_list[i], label=label[i], color=color[i], linestyle=line_style[i])
    plt.plot(theta_desire_list[0], label="$\\theta_{target}$", linestyle="--")
    plt.legend(loc='best', prop={'family': 'Times New Roman'})
    # plt.title("$ \\theta$  Control Curve With Filter PI2", fontdict={'family': 'Times New Roman'})
    save_figure("./photo/exp1/", "theta_Curve.pdf")
    plt.show()
def plot_loss_k(K_after_training_list ,loss_after_training_list,train_time,figure_number = 3):

    label = ["FR-PI2","F-PI2","PI2","Reference parameter"]
    color = ["r", "g", "b", "k"]
    line_style = ["-", "--", ":", "-."]
    marker = ['*', '^', 'h']

    # "绘制Kp曲线"
    # plt.xticks(fontproperties='Times New Roman')
    # plt.yticks(fontproperties='Times New Roman')
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("$K_p$ Value")
    # print("FUCK ",train_time)
    # for i in range(figure_number):
    #         plt.plot(K_after_training_list[i][0][:train_time], label=label[i], color=color[i],
    #                  linestyle=line_style[i],marker=marker[i])
    # plt.legend(loc='best', prop={'family': 'Times New Roman'})
    # # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混
    # plt.title("$K_p$ Iteration Graph", fontdict={'family': 'Times New Roman'})
    # save_figure("./photo/exp1/", "Kp_curve.pdf")
    # plt.show()
    # "绘制Kd曲线"
    # plt.xticks(fontproperties='Times New Roman')
    # plt.yticks(fontproperties='Times New Roman')
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("$K_p$ Value")
    # for i in range(figure_number):
    #     plt.plot(K_after_training_list[i][1][:train_time], label=label[i], color=color[i],
    #              linestyle=line_style[i],marker=marker[i])
    # plt.legend(loc='best', prop={'family': 'Times New Roman'})
    # # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混
    # plt.title("$K_d$ Iteration Graph", fontdict={'family': 'Times New Roman'})
    #
    # save_figure("./photo/exp1/", "Kd_curve.pdf")
    # plt.show()
    # "绘制Ki曲线"
    # plt.xticks(fontproperties='Times New Roman')
    # plt.yticks(fontproperties='Times New Roman')
    # plt.xlabel("Number of Iterations")
    # plt.ylabel("$K_i$ Value")
    # for i in range(figure_number):
    #     plt.plot(K_after_training_list[i][2][:train_time], label=label[i], color=color[i],
    #              linestyle=line_style[i],marker=marker[i])
    # plt.legend(loc='best', prop={'family': 'Times New Roman'})
    # # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混
    # plt.title("$K_i$ Iteration Graph", fontdict={'family': 'Times New Roman'})
    # save_figure("./photo/exp1/", "Ki_curve.pdf")
    # plt.show()





    "绘制LOSS曲线"
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Number of Iterations",fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("Loss",fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(loss_after_training_list[i][:train_time], label=label[i], color=color[i],
                     linestyle=line_style[i], marker=marker[i],markersize=markersize,linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': fontsize})
    # plt.title("Loss Function Curve With Natural PI2", fontdict={'family': 'Times New Roman'})
    save_figure("./photo/exp1/", "loss.pdf")
    plt.show()
if __name__ == "__main__":
    ## 老师的路径积分
    first_time =time.time()
    poll = mp.Pool(mp.cpu_count())

    reward_model = PID_model()
    alpha_list =[]
    delta_z_list = []
    theta_list = []
    theta_desire_list =[]
    k_list = []
    loss_list = []
    maxtime =-1
    k = np.zeros((3,1),dtype=np.float64)

    training_times = 10
    model = RL_PI2(if_filter=True, attenuation_step_length=1, alpha=0.85)
    model.set_initial_value([100, 100, 100])
    alpha, delta_z, theta, theta_desired, k_after, loss, cur_time = model.rolling_optimization(rolling_time=20,
                                                                                               total_step=200)
    alpha_list.append(alpha)
    delta_z_list.append(delta_z)
    theta_list.append(theta)
    theta_desire_list.append(theta_desired)
    k_list.append(k_after)
    loss_list.append(loss)
    maxtime = max(maxtime, cur_time)
    training_times = maxtime
    # # ## 第1组优化10次
    model = RL_PI2(if_filter=True,attenuation_step_length=1,alpha=0.85)
    model.set_initial_value([100, 100, 100])
    k[0],k[1],k[2],k_after,loss,cur_time = model.training()
    alpha, delta_z, theta, theta_desired = reward_model.model_simulation(k[0],k[1],k[2],200)
    alpha_list.append(alpha)
    delta_z_list.append(delta_z)
    theta_list.append(theta)
    theta_desire_list.append(theta_desired)
    k_list.append(k_after)
    loss_list.append(loss)
    # training_times = maxtime
    # # ## 第2组优化10次
    model = RL_PI2(if_filter=False,attenuation_step_length=1,alpha=1/0.85)
    model.set_initial_value([100, 100, 100])
    k[0], k[1], k[2], k_after, loss, cur_time = model.training()
    alpha, delta_z, theta, theta_desired = reward_model.model_simulation(k[0], k[1], k[2],200)
    alpha_list.append(alpha)
    delta_z_list.append(delta_z)
    theta_list.append(theta)
    theta_desire_list.append(theta_desired)
    k_list.append(k_after)
    loss_list.append(loss)
    maxtime = max(maxtime, cur_time)
    # ## 第2组优化10次
    print(maxtime)
    # ## 第2组优化10次
    #     # model = RL_PI2(if_filter=False)
    #     # model.set_initial_value([100, 100, 100])
    #     # k[0], k[1], k[2], k_after, loss, cur_time = model.training()
    #     # alpha, delta_z, theta, theta_desired = reward_model.model_simulation(k[0], k[1], k[2])
    #     # alpha_list.append(alpha)
    #     # delta_z_list.append(delta_z)
    #     # theta_list.append(theta)
    #     # theta_desire_list.append(theta_desired)
    #     # k_list.append(k_after)
    #     # loss_list.append(loss)
    #     # maxtime = max(maxtime, cur_time)
    # ## 第2组仅仅优化一次
    # model.set_initial_value()
    # alpha, delta_z, theta, theta_desired = model.rolling_optimization(rolling_time=500, total_step=500)
    # alpha_list.append(alpha)
    # delta_z_list.append(delta_z)
    # theta_list.append(theta)
    # theta_desire_list.append(theta_desired)
    # # ## 第3组对照组
    # # alpha, delta_z, theta, theta_desired = reward_model.model_simulation(1.5,2.5,0.5,1000)
    # # alpha_list.append(alpha)
    # # delta_z_list.append(delta_z)
    # # theta_list.append(theta)
    # # theta_desire_list.append(theta_desired)
    # ## 第3组对照组
    # alpha, delta_z, theta, theta_desired = reward_model.model_simulation(Init_K[0],Init_K[1],Init_K[2], 500)
    # alpha_list.append(alpha)
    # delta_z_list.append(delta_z)
    # theta_list.append(theta)
    # theta_desire_list.append(theta_desired)
    ## 绘图
    plot_result(alpha_list,delta_z_list,theta_list,theta_desire_list,figure_number=3)
    plot_loss_k(k_list, loss_list, maxtime,figure_number=3)