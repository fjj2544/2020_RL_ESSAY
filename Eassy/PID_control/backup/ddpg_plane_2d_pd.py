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
#定义环境,参数已经设置完全，参数设置好了。动力学模型已经建立，主要就是写算法
class Planes_Env:
    def __init__(self):
        # self.actions = [0,1]
        self.observation_dim = 2
        #状态量分别为[飞机迎角 alpha, 飞机俯仰角theta, 飞机俯仰角速度q]
        self.state = np.array([0.0, 0.0, 0.0])
        self.observation = np.array([0.0,0.0])
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
        #角度阈值 攻角alpha[-1,10];舵偏delta_z[-20,20]
        self.alpha_threshold_max = 10
        self.alpha_threshold_min = -1
        self.delta_z_threhold_max = 20
        self.delta_z_threhold_min = -20
        self.reward = 0.0
        self.cost = 0.0
    def reset(self):
        n = np.random.randint(1,1000,1)
        np.random.seed(n)
        self.state = np.random.uniform(-0.5, 0.5,size=(3,) )
        # print(self.state)
        # self.state = np.array([0.0,0.0,0.0])
        self.observation = np.array([0.0,0.0])
        # self.state = np.array([0.0, 5.0, 0.0])
        self.steps_beyond_done = 0
        # print(self.state)
        return self.observation
    def step(self,action):
        # print("action", action)
        action = action[0]
        state = self.state
        alpha,theta, q= state
        observation_pre = theta-self.theta_desired
        delta_z = action
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        #动力学方程 攻角alpha，俯仰角theta 俯仰角速度q  舵偏delta_z
        alpha_dot = q - self.b_alpha * alpha - self.b_delta_z * delta_z
        theta_dot = q
        q_dot = -self.a_alpha*alpha - self.a_q*q - self.a_delta_z*delta_z
        #积分得到状态量
        q = q + self.tau * q_dot
        theta = theta + self.tau * theta_dot
        observation_cur = theta - self.theta_desired
        alpha = alpha + self.tau *alpha_dot
        self.steps_beyond_done += 1
        self.state = np.array([alpha, theta, q])
        #根据更新的状态判断是否结束
        lose = alpha < self.alpha_threshold_min or alpha > self.alpha_threshold_max
        # done = bool(done)
        #设置回报
        if not lose :
            self.reward =-((theta-self.theta_desired)**2+0.1*q**2+0.01*action**2)
            # self.steps_beyond_done = self.steps_beyond_done+1
        # elif self.steps_beyond_done is None:
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        else:
            self.reward = -2500
        done = lose or self.steps_beyond_done > self.max_steps
        # print('cost', self.cost)
        # print(-self.cost)
        self.observation = np.array([observation_pre,observation_cur])
        return self.observation, self.reward, done

action_bound = [-20, 20]

# PID模型已经解决
## 简单的pid训练的方法
class PID_model():
    def __init__(self):
        self.env = Planes_Env()
        pass
    def get_epsolid_reward(self,k1=1.5,k2=2.5,k3=0.5):
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
        # 峰值时间
        tp =0
        ## 调整时间
        ts = 0
        while i < total_step:
            error = self.env.theta_desired - self.env.state[1]
            derror = -self.env.state[2]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror
            if(error == 0 and tp ==0):
                tp = i
            self.env.step(np.array([action]))
            alpha.append(self.env.state[0])
            theta.append(self.env.state[1])
            desired_theta.append(self.env.theta_desired)
            q.append(self.env.state[2])
            time.append(i)
            control.append(action)
            i = i + 1

        return abs(max(theta)-max(desired_theta))


    def model_simulation(self,k1=1.5,k2=2.5,k3=0.5,iterator =0 ):
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
        while i < total_step:
            error = self.env.theta_desired - self.env.state[1]
            derror = -self.env.state[2]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror
            self.env.step(np.array([action]))
            alpha.append(self.env.state[0])
            theta.append(self.env.state[1])
            desired_theta.append(self.env.theta_desired)
            q.append(self.env.state[2])
            time.append(i)
            control.append(action)
            i = i + 1
        """
        调用签名：plt.plot(x, y,  ls="-", lw=2, label="plot figure")

        x: x轴上的数值
        
        y: y轴上的数值
        
        ls：折线图的线条风格
        
        lw：折线图的线条宽度
        
        label：标记图内容的标签文本 必须配合legend使用
        """
        plt.figure(2)
        plt.plot(time, theta,label="time-theta")
        plt.plot(time, desired_theta,label="time-desired_theta")
        plt.legend(loc="best")
        plt.title("This is the %s epoch" % str(iterator))
        plt.show()
        # plt.figure(2)
        # plt.plot(time,derror_list,label="time-derror")
        # plt.plot(time,error_list,label="time-error")
        # plt.legend(loc="best")
        # print(max(derror_list),min(derror_list))
        # print(max(error_list),min(error_list))
        plt.show()
## 路径积分的方法
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
1. 这个方法有问题
2. 很容易收敛到一个缺点
3. 离散化没有做
4. 全靠方差进行探索可能有问题
'''
## 暂时可以用这个尝试否则无法离散化
## 离散化只不过是离散化初始点的动作，如果从这个动作开始应该以怎么样的参数继续能达到更好地效果
'''
这里我想用路径积分的方法解决这个问题
这里的环境自己的定义， 
'''


error_bound = [-10,10]
""" -----------------------------------------------------------代码1-------------------------------------------------------------"""
def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma;
    return x

class Path_Integral():
    def __init__(self):
        # 环境
        self.env = Planes_Env()
        # 轨迹生成器,得到稀疏回报
        self.PID_model = PID_model()
        # 最初的K向量
        self.K = np.array([1.5,2.5,0.5])
        # 噪声方差，讲道理这里应该是一个协方差矩阵这个最开始就不能很大,但是你也不能没有探索，否则局部最优
        ## 这个参数一定不能太大
        self.sigma = 0.5
        # 策略迭代次数
        self.M = int(1e4)
        # 每次策略评估次数
        self.N = int(1e2)
        # 衰减频率
        self.ma = 10
        # 学习率,用于衰减探索量
        self.learning_rate = 0.85
        # 如果确实越来越好，我可以每过一段时间输出一个图片看看
    def update_k(self,reward_list,eplison_list):
        # 计算概率分母
        sum = 0.
        p_list = []
        for reward in reward_list:
            sum += math.exp(-reward)
        # 计算概率分子
        for reward in reward_list:
            p_list.append( math.exp(-reward)/sum)
        # 对于所有变化求和进行筛选
        for i in range(len(p_list)):
            eplison = p_list[i] * reward_list[i]
        return eplison
    def train(self):
        ## 策略评估然后策略改进
        # 存储最优迭代的K，初始值就是基准值
        plot_reward_list =[]
        best_k_vector = self.K
        for iterator in range(self.M):
            # 存每一幕的稀疏回报
            reward_list = []
            # 存储每一幕的高斯噪声
            eplison_list = []
            # 策略评估
            for i in range(self.N):
                ## 在基础k向量上面加噪声然后把噪声存储起来
                # 我觉得应该加入一个不变的
                if i != 0:
                    temp_k_vector  =  best_k_vector[:]+random.gauss(0,self.sigma)
                else:
                    # 我个人认为，为了算法稳定，我一定要添加当前的参数进入学习序列，否则会有很大波动
                    temp_k_vector = best_k_vector
                # 每一幕的高斯噪声
                eplison = temp_k_vector -  best_k_vector
                # 这里是一个array的list,两个完全不同
                eplison_list.append(eplison)
                # 交互得到回报
                r = self.PID_model.get_epsolid_reward(temp_k_vector[0],temp_k_vector[1],temp_k_vector[2])
                # 存储回报
                reward_list.append(r)
            # 结果可视化,查看是否越变越好
            print(best_k_vector, np.mean(reward_list))
            plot_reward_list.append(np.mean(reward_list))
            # 0-1标准化回报，标准化回报很关键
            reward_list = preprocessing.scale(reward_list)
            # 下面这玩意肯定是不行了
            # reward_list = preprocessing.minmax_scale(reward_list)
            # 策略改进
            best_k_vector =  best_k_vector +  self.update_k(reward_list,eplison_list)
            if iterator % self.ma ==0:
                self.sigma = self.learning_rate * self.sigma
                self.PID_model.model_simulation(best_k_vector[0],best_k_vector[1],best_k_vector[2],iterator)
                plt.plot(plot_reward_list,label="reward")
                plt.title("This is the %s epoch"%str(iterator))
                plt.legend(loc="best")
                plt.show()




""" -----------------------------------------------------------代码2-------------------------------------------------------------"""
# 训练次数,也就是策略迭代次数
training_times = 100  # training times
# 路径数量,也就是策略评估数量
roll_outs = 20  # path number


class RL_PI2:
    def __init__(self):
        # 最开始的K
        self.K = np.zeros((3, 1), dtype=np.float64)
        # 记录每次策略评估中的K
        self.K_roll = np.zeros((3, roll_outs), dtype=np.float64)  # training parameters
        # 记录策略迭代中全部的K,也就是每次策略评估中的K都会被记录下来
        self.K_record = np.zeros((3, roll_outs, training_times), dtype=np.float64)  # record training parameters
        # 噪声，也就是两个的方差不一样
        self.sigma = np.zeros((3, 1), dtype=np.float64)  # standard deviation of the variety about training parameters
        # 每一次噪声给数据变化了多少
        self.k_delta = np.zeros((3, roll_outs), dtype=np.float64)
        # 也就是记录每一局的回报
        self.loss = np.zeros((roll_outs, 1), dtype=np.float64)  # loss function
        # 记录所有策略迭代中的回报
        self.loss_record = np.zeros((roll_outs, training_times), dtype=np.float64)  # record loss function
        # 也就是每次迭代完之后计算一下回报
        self.loss_after_training = np.zeros((training_times, 1), dtype=np.float64)  # loss function after each training
        # 每次迭代都记录参数
        self.K_after_training = np.zeros((3, training_times), dtype=np.float64)  # K after each training

        self.alpha = 0  # attenuation coefficient
        # 衰减频率
        self.attenuation_step_length = 10  # sigma is attennuated every attenuation_step_length training times
        # 衰减参数
        self.alpha = 0.85  # sigma is attennuated at 0.85 ratio
        # 当前roll也就是第几幕
        self.current_roll = 0
        # 当前迭代次数,也就是第几次训练
        self.current_training = 0
        # PI2系数
        self.PI2_coefficient = 30.0  # PI2 coefficient
        ## 交互模型
        self.reward_model = PID_model()


    def sample_using_PI2(self):  # using PI2 to sample
        time_start = time.time()
        print('start')
        self.set_initial_value()
        # 直接训练
        self.training()  # start training

        time_end = time.time()
        print('end')
        print('total time', time_end - time_start)
        # np.savetxt('k_single_one.txt', self.loss_after_training)

    def set_initial_value(self):
        self.K[0] = 1.5
        self.K[1] = 2.5
        self.K[2] = 0.5
        ## 延迟更新，基于调节PID参数的经验，首先调节sigma然后衰减，然后调节kd，然后衰减
        self.sigma[0] = 1.0
        self.sigma[1] = 0.3
        self.sigma[2] = 0.1

    # 训练过程
    def training(self):
        for i in range(training_times):
            self.current_training = i
            # 直接进行衰减
            if i % self.attenuation_step_length == 0 and i != 0:
                self.sigma = self.sigma * self.alpha  # attenuation
                self.reward_model.model_simulation(self.K[0],self.K[1],self.K[2],i)
            # 采样N局,记录稀疏回报
            for j in range(roll_outs):
                self.current_roll = j
                self.k_delta[0, j] = np.random.normal(0, self.sigma[0], 1)
                self.k_delta[1, j] = np.random.normal(0, self.sigma[1], 1)
                self.k_delta[2, j] = np.random.normal(0, self.sigma[2], 1)

                print(self.k_delta[:,j])
                self.K_roll[0, j] = self.K[0] + self.k_delta[0, j]
                self.K_roll[1, j] = self.K[1] + self.k_delta[1, j]
                self.K_roll[2, j] = self.K[2] + self.k_delta[2, j]
                self.loss[j] = self.reward_model.get_epsolid_reward(self.K_roll[0,j],self.K_roll[1,j],self.K_roll[2,j])
                # 为啥这里要防止同样的回报
                self.loss[j] = self.loss[j] + np.random.uniform(-0.02, 0.02, 1)
            self.K_record[:, :, self.current_training] = self.K_roll
            self.loss_record[:, self.current_training] = self.loss[:, 0]
            exponential_value_loss = np.zeros((roll_outs, 1), dtype=np.float64)  #
            probability_weighting = np.zeros((roll_outs, 1), dtype=np.float64)  # probability weighting of each roll
            # 果然这里要做一个max min 标准化
            for i2 in range(roll_outs):
                exponential_value_loss[i2] = np.exp(-self.PI2_coefficient * (self.loss[i2] - self.loss.min())
                                                    / (self.loss.max() - self.loss.min()))
            for i2 in range(roll_outs):
                probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)

            temp_k = np.dot(self.k_delta, probability_weighting)
            # print(self.sigma)

            # updata
            self.K = self.K + temp_k
            self.K_after_training[:, self.current_training] = self.K[:, 0]
            self.loss_after_training[self.current_training] = self.reward_model.get_epsolid_reward(self.K[0],self.K[1],self.K[2])
        plt.plot(self.loss_after_training)
        plt.show()

if __name__=="__main__":
    ### 简单PID测试
    # model  = PID_model()
    # model.model_simulation()
    # ### 路径积分测试
    # model =Path_Integral()
    # model.train()
    ## 老师的路径积分
    model = RL_PI2()
    model.sample_using_PI2()






## 算法很不稳定
## 算法无法控制
## 不确定改变的方向
## 很容易一下子就变得好，然后又回去了。
# 跟设置关系非常大
# 没有办法解决负偏差