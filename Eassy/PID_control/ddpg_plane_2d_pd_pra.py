import pygame
import numpy as np
import tensorflow as tf
# from load import *
from pygame.locals import *
import math
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing

RENDER = False
import random
import multiprocessing as mp
import numba
from numba import jit

C_UPDATE_STEPS = 10
A_UPDATE_STEPS = 1
"""
空气动力学模型
"""


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
# 定义环境,参数已经设置完全，参数设置好了。动力学模型已经建立，主要就是写算法
action_bound = [-20, 20]
alpha_bound = [-1,10]

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

    def reset(self):
        n = np.random.randint(1, 1000, 1)
        np.random.seed(n)
        self.state = np.random.uniform(-0.5, 0.5, size=(3,))
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        return self.observation

    def step(self, action):
        action = action[0]
        state = self.state
        alpha, theta, q = state
        observation_pre = theta - self.theta_desired
        ## 非线性约束
        self.delta_z = np.clip(action,self.delta_z_threhold_min,self.delta_z_threhold_max)



        # 动力学方程 攻角alpha，俯仰角theta 俯仰角速度q  舵偏delta_z
        alpha_dot = q - self.b_alpha * alpha - self.b_delta_z * self.delta_z
        theta_dot = q
        q_dot = -self.a_alpha * alpha - self.a_q * q - self.a_delta_z * self.delta_z
        # 积分得到状态量
        q = q + self.tau * q_dot


        theta = theta + self.tau * theta_dot
        observation_cur = theta - self.theta_desired
        alpha = np.clip(alpha + self.tau * alpha_dot,self.alpha_threshold_min,self.alpha_threshold_max)


        self.steps_beyond_done += 1
        self.state = np.array([alpha, theta, q])
        # 根据更新的状态判断是否结束
        lose = alpha < self.alpha_threshold_min or alpha > self.alpha_threshold_max
        # 设置回报
        if not lose:
            self.reward = -((theta - self.theta_desired) ** 2 + 0.1 * q ** 2 + 0.01 * action ** 2)
        else:
            self.reward = -2500
        done = lose or self.steps_beyond_done > self.max_steps
        self.observation = np.array([observation_pre, observation_cur])
        return self.observation, self.reward, done


''' 要有行之有效的方法,能够解决一类自动整定问题 随机设置一个初始点都能收敛到一个较好的点'''
## 目标量,采用动态目标
Overshoot_target = 1e-3
ts_target = 150
Waveform_oscillation_bound = 1
Static_error_bound = 0.01
## 调整时间的计算范围,连续K次测试,保证K次测试内能通过
adjust_bound = 0.02
#可信度
belief_times = 50


# PID模型已经解决
## 简单的pid训练的方法
class PID_model():
    def __init__(self):
        self.env = Planes_Env()

    def get_epsolid_reward(self, k1=1.5, k2=2.5, k3=0.5,is_test = False):
        total_step = 2000
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

        dez_list = []
        # 峰值时间
        tp = 0
        '''计算调整时间   如果调整时间过大我们就加大惩罚'''
        ts = total_step
        count = 0
        for i in range(total_step):
            if count >= belief_times and ts == total_step:
                ts = i
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


        #  强调稳态误差，不等权reward,可以减少稳态误差


        ### 尺度不一样就很难解决问题

        ## 数据尺度归一化很重要
        # 超调量 kp
        Overshoot= abs(max(abs(np.array(theta))) - max(abs(np.array(desired_theta))))
        # 静态误差 ki
        Static_error = abs(error_list[-1])
        # 波形震荡  kd
        Waveform_oscillation = np.var(theta)
        # 调整时间
        weight = 100
        # 动作震荡
        action_oscillation =np.var(dez_list)

        ts = 0 if ts<=ts_target else ts
        # r =  weight/Overshoot_boundOvershoot + weight/Static_error_bound*Static_error + weight/Waveform_oscillation_bound*Waveform_oscillation +ts
        r = weight/Overshoot_target*Overshoot + weight*ts/ts_target

        # 判断是否满足约束,约束判准
        if is_test:
            return ts
        else:
            return r


    def model_simulation(self, k1=1.5, k2=2.5, k3=0.5, iterator=0):
        total_step = 2000
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

            # np.savetxt()
        """
        调用签名：plt.plot(x, y,  ls="-", lw=2, label="plot figure")

        x: x轴上的数值

        y: y轴上的数值

        ls：折线图的线条风格

        lw：折线图的线条宽度

        label：标记图内容的标签文本 必须配合legend使用
        """
        plt.plot(alpha,label="alpha")
        plt.legend(loc="best")
        plt.title("This is the %s epoch alpha" % str(iterator))
        plt.show()
        plt.plot(dez_list, label="real_dez")
        plt.legend(loc="best")
        plt.title("This is the %s epoch ACTION-dez" % str(iterator))
        plt.show()
        plt.figure(2)
        plt.plot(time, theta, label="time-theta")
        plt.plot(time, desired_theta, label="time-desired_theta")
        plt.legend(loc="best")
        plt.title("This is the %s epoch" % str(iterator))
        plt.savefig("%sepoch.png"%iterator)
        plt.show()


        # plt.figure(2)
        # plt.plot(time,derror_list,label="time-derror")
        # plt.plot(time,error_list,label="time-error")
        # plt.legend(loc="best")
        # print(max(derror_list),min(derror_list))
        # print(max(error_list),min(error_list))
        plt.show()



""" -----------------------------------------------------------算法伪代码-------------------------------------------------------------"""

'''
首先参数化控制策略就是简单的PID控制
离散化状态空间 ed e ei
Parameterize the control using (15);
Discretize the state space using (18);
for each initial state:  这个地方指的是 state = (e,de,ie)
    initiate the parameter vector K;
    initiate the Gaussian variance Σ;
    for iteration = 1, …, M:
        for i = 1, …, N:
            generate a path τi using Gaussian noise εi with
            variance Σ;
            obtain the path loss S (τi ) using (14);
        end for
        update parameter vector K using (13) and (17);
        if iteration%Mα == 0:
            attenuate Gaussian variance Σ using (19);
        end if
    end for
end for

'''
## 暂时可以用这个尝试否则无法离散化
## 离散化只不过是离散化初始点的动作，如果从这个动作开始应该以怎么样的参数继续能达到更好地效果
'''
这里我想用路径积分的方法解决这个问题
这里的环境自己的定义， 
'''

error_bound = [-10, 10]
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
# 最大调整次数, 分为两个阶段 首先降低超调、振幅、静态误差 然后进行约束最优化调整 ，这个想法来自于分层学习和迁移学习以及我们的调半个学期 P I D的经验
# 后期可能加入延迟更新
Max_Adjust_times = 100
#测试次数,也就是测试可信度
Test_Times = 10
class RL_PI2:
    def __init__(self):
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
        """ -----------------------------------------------------------定义算法超参数-------------------------------------------------------------"""
        # 衰减频率
        self.attenuation_step_length = 10
        # 衰减系数 这里我觉得可以加入先验知识,不等权衰减
        self.alpha = 0.85
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
        ## 是否记录图片
        self.save_photo = True

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
        self.training()

        time_end = time.time()
        print('end')
        print('total time', time_end - time_start)

        if self.save_data:
            self.data_record()

    def set_initial_value(self):

        if self.random_init:
            self.K[0] = random.uniform(Kp_Min,Kp_Max)
            self.K[1] = random.uniform(Kd_Min,Kd_Max)
            self.K[2] = random.uniform(Ki_Min,Ki_Max)
        else:
            self.K[0] = 1.5
            self.K[1] = 2.5
            self.K[2] = 0.5
        # 初始化方差
        self.sigma[0] = 1.0
        self.sigma[1] = 0.3
        self.sigma[2] = 0.1
        # 初始化记录参数
        self.current_roll = 0
        self.current_training = 0

    """ -----------------------------------------------------------计算轨迹回报,用于并行------------------------------------------------------------"""
    @jit(forceobj=True,nopython=True,nogil=True)
    def cal_eposlid_loss(self,j):
        self.current_roll = j
        delta1 = np.random.normal(0, self.sigma[0], 1)
        delta2 = np.random.normal(0, self.sigma[1], 1)
        delta3 = np.random.normal(0, self.sigma[2], 1)



        cur_k1 = self.K[0] + delta1
        cur_k2 = self.K[1] + delta2
        cur_k3 = self.K[2] + delta3
        loss = self.reward_model.get_epsolid_reward(cur_k1, cur_k2, cur_k3)
        ##如果没有优势,那么我们就可以不去学习,没有必要浪费时间去学习没有用的东西
        if (self.current_training > 1 and loss > self.loss_after_training[self.current_training - 1]):
            delta1 = delta2 = delta3 = 0.0
            cur_k1 = self.K[0] + delta1
            cur_k2 = self.K[1] + delta2
            cur_k3 = self.K[2] + delta3
        return delta1,delta2,delta3,cur_k1,cur_k2,cur_k3,loss
    """ -----------------------------------------------------------策略评估------------------------------------------------------------"""
    @jit(forceobj=True, nopython=True, nogil=True,parallel=True)
    def policy_evl(self):
        # 采样N局,记录稀疏回报
        multi_res = [poll.apply_async(self.cal_eposlid_loss, (j,)) for j in range(roll_outs)]
        for j, res in enumerate(multi_res):
            self.k_delta[0, j] = res.get()[0]
            self.k_delta[1, j] = res.get()[1]
            self.k_delta[2, j] = res.get()[2]
            self.K_roll[0, j] = res.get()[3]
            self.K_roll[1, j] = res.get()[4]
            self.K_roll[2, j] = res.get()[5]
            self.loss[j] = res.get()[6]
            self.loss[j] = self.loss[j] + np.random.uniform(-0.02, 0.02, 1)
            # print(self.k_delta,self.K_roll,self.loss)

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

    """ ----------------------------------------------------------校验是否满足约束 用于并行------------------------------------------------------------"""
    def test_res(self):
        if self.reward_model.get_epsolid_reward(self.K[0], self.K[1], self.K[2], True):
          return True
        else:
            return False
    def Test_K(self):
        flag = True
        multi_res = [poll.apply_async(self.test_res, ()) for j in range(Test_Times)]
        for res in multi_res:
            flag &= res.get()
        if flag :
            return True
        else:
            return False
    """ ----------------------------------------------------------策略迭代部分------------------------------------------------------------"""
    def training(self):
        i =0
        adjust_times = 0
        while i < training_times+adjust_times:
            # 分阶段优化 首先调整到局部最优 然后找到带有约束的满意解
            self.current_training = i
            # 方差衰减和可视化
            if self.current_training % self.attenuation_step_length == 0  :
                self.sigma = self.sigma / self.alpha  # attenuation
                self.reward_model.model_simulation(self.K[0], self.K[1], self.K[2],self.current_training)
            # 策略迭代框架
            self.policy_evl()
            self.policy_improve()
            # 记录参数
            self.K_after_training[:, self.current_training] = self.K[:, 0]
            self.loss_after_training[self.current_training] = self.reward_model.get_epsolid_reward(self.K[0], self.K[1],
                                                                           self.K[2])
            """ ----------------------------------------------------------虽然可以用但是这一块有BUG------------------------------------------------------------"""
            # 输出当前训练次数
            if(self.current_training % self.attenuation_step_length == 0 ):
                print(self.current_training,time.time()-first_time)
            i+=1


        plt.plot(self.K_after_training[0][:training_times+adjust_times],label="KP")
        plt.plot(self.K_after_training[1][:training_times+adjust_times],label="KD")
        plt.plot(self.K_after_training[2][:training_times+adjust_times],label="KI")
        plt.legend(loc="best")
        plt.savefig("K.png")
        plt.show()
        plt.plot(self.loss_after_training[:training_times+adjust_times])
        plt.savefig("loss.png")
        plt.show()

        ##最后输出依次
        self.reward_model.model_simulation(self.K[0],self.K[1],self.K[2],self.current_training)
        print(self.reward_model.get_epsolid_reward(self.K[0],self.K[1],self.K[2],True))

if __name__ == "__main__":
    ### 简单PID测试
    # model  = PID_model()
    # model.model_simulation()
    # ### 路径积分测试
    # model =Path_Integral()
    # model.train()
    ## 老师的路径积分

    first_time =time.time()
    poll = mp.Pool(mp.cpu_count())
    model = RL_PI2()
    model.sample_using_PI2()



"""
1. 就学习好的部分，消除差的部分，就不管理 ,如果比上一步的回报更差 
2. 探索性初始化, 全局网格搜索 
3. reward 存在很多鞍点 
4. 两步优化 , 滚动优化 , 优点麻烦
"""
