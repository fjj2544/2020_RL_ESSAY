# -*- coding: UTF-8 -*-

import numpy as np
import time
import matplotlib.pyplot as pl

# 训练次数,也就是策略迭代次数
training_times = 100  # training times
# 路径数量,也就是策略评估数量
roll_outs = 20  # path number
# 仿真步长
dt = 0.035  #step length of simulation, second
# 最大仿真时间
max_step = 450  #maximum time steps

N_d = 161
N_phi = 35

class RL_PI2:
    def __init__(self):
        # 最开始的K
        self.K = np.zeros((2, 1),dtype=np.float64)
        # 记录每次策略评估中的K
        self.K_roll = np.zeros((2, roll_outs), dtype=np.float64)  #training parameters
        # 记录策略迭代中全部的K,也就是每次策略评估中的K都会被记录下来
        self.K_record = np.zeros((2, roll_outs, training_times), dtype=np.float64)  # record training parameters
        # 噪声，也就是两个的方差不一样
        self.sigma = np.zeros((2, 1), dtype=np.float64)  #standard deviation of the variety about training parameters
        # 每一次噪声给数据变化了多少
        self.k_delta = np.zeros((2, roll_outs), dtype=np.float64)
        # 也就是记录每一局的回报
        self.loss = np.zeros((roll_outs, 1), dtype=np.float64)  #loss function
        # 记录所有策略迭代中的回报
        self.loss_record = np.zeros((roll_outs, training_times), dtype=np.float64)  #record loss function
        # 也就是每次迭代完之后计算一下回报
        self.loss_after_training = np.zeros((training_times, 1), dtype=np.float64)  #loss function after each training
        # 每次迭代都记录参数
        self.K_after_training = np.zeros((2, training_times), dtype=np.float64)  # K after each training

        self.alpha = 0  # attenuation coefficient
        # 衰减频率
        self.attenuation_step_length = 10  #sigma is attennuated every attenuation_step_length training times
        # 衰减参数
        self.alpha = 0.85  #sigma is attennuated at 0.85 ratio
        # 当前roll也就是第几幕
        self.current_roll = 0
        # 当前迭代次数,也就是第几次训练
        self.current_training = 0
        # PI2系数
        self.PI2_coefficient = 30.0 #PI2 coefficient

        ###动力学模型
        self.u = 0.22   #m/s
        self.r = 0    #rad/s
        self.r_last = 0 #rand/s
        self.max_r = 1.58   #rad/s
        self.max_r_delta = 1.58   #rad/s

        # 初始状态,下面做了一个离散化
        self.d = 0
        self.phi = 0

        ### 这个离散化后面基本没有用到,也就是现在还是求单点的最优
        self.sample_d = np.zeros((N_d, 1), dtype=np.float64)
        self.sample_phi = np.zeros((N_phi, 1), dtype=np.float64)
        for i in range(N_d):
            self.sample_d[i] = i / 100 - 0.8
        for i in range(N_phi):
            self.sample_phi[i] = i * 5.0 / 180.0 * np.pi - 85.0 / 180.0 * np.pi

    def sample_using_PI2(self):  # using PI2 to sample
        time_start = time.time()
        print('start')
        # 初始化
        self.set_initial_value(-0.61, -55/180.0*np.pi)
        # 直接训练
        self.training()  #start training

        time_end = time.time()
        print('end')
        print('total time', time_end - time_start)
        np.savetxt('k_single_one.txt',self.loss_after_training)

    def set_initial_value(self, d_init, phi_init):
        self.K[0] = 30.0
        self.K[1] = 5.0
        self.sigma[0] = 2.0
        self.sigma[1] = 0.8

        self.d = d_init  # distance error, m
        self.phi = phi_init  # orientation error, rad
    # 训练过程
    @jit
    def training(self):
        for i in range(training_times):
            self.current_training = i
            # 直接进行衰减
            if i % self.attenuation_step_length == 0 and i!=0:
                self.sigma = self.sigma  * self.alpha  # attenuation
            # 采样N局,记录稀疏回报
            for j in range(roll_outs):
                self.current_roll = j
                self.k_delta[0, j] = np.random.normal(0, self.sigma[0], 1)
                self.k_delta[1, j] = np.random.normal(0, self.sigma[1], 1)
                self.K_roll[0, j] = self.K[0] + self.k_delta[0, j]
                self.K_roll[1, j] = self.K[1] + self.k_delta[1, j]
                self.loss[j] = self.model(self.K_roll[0, j], self.K_roll[1, j])
                # 为啥这里要防止同样的回报
                self.loss[j] = self.loss[j] + np.random.uniform(-0.02, 0.02, 1)
            #
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

            #updata
            self.K = self.K + temp_k
            self.K_after_training[:, self.current_training] = self.K[:, 0]
            self.loss_after_training[self.current_training] = self.model(self.K[0], self.K[1])

        pl.plot(self.loss_after_training)
        pl.show()
    ## 训练终止条件
    def flag_stop(self, d0, phi0):
        flag = 0
        flag1 = np.zeros((2, 10), dtype=np.float64)
        if np.abs(d0[0]) < 0.001:
            flag1[0, 0] = 1
        if np.abs(phi0[0]) < 0.5 / 180.0 *np.pi:
            flag1[1, 0] = 1
        for i3 in range(9):
            if np.abs(d0[i3 + 1] - d0[0]) < 0.001:
                flag1[0, i3 + 1] = 1
            if np.abs(phi0[i3 + 1] - phi0[0]) < 0.5 / 180.0 * np.pi:
                flag1[1, i3 + 1] = 1
        for i3 in range(2):
            for j3 in range(10):
                if flag1[i3, j3] == 1:
                    flag = flag + 1
        if flag == 20:
            return 1
        else:
            return 0
    # 动力学模型,也就是我们的PID模型,这里是一个李亚普洛夫函数
    def model(self, K0, K1):
        d_temp = np.zeros((10, 1), dtype=np.float64)
        phi_temp = np.zeros((10, 1), dtype=np.float64)
        my_d = self.d
        my_phi = self.phi
        d_temp[9] = my_d
        phi_temp[9] = my_phi
        for tt in range(max_step):
            self.r =  -K0 * self.u * my_d * np.sin(my_phi) / my_phi - K1 * np.abs(self.u) * my_phi
            #constraint
            if self.r > self.max_r:
                self.r = self.max_r
            if self.r - self.r_last > self.max_r_delta:
                self.r = self.r_last + self.max_r_delta
            if self.r < -self.max_r:
                self.r = -self.max_r
            if self.r - self.r_last < -self.max_r_delta:
                self.r = self.r_last - self.max_r_delta
            #updata
            my_d = my_d + self.u * dt * np.sin(my_phi)
            my_phi =my_phi + self.r * dt
            self.r_last = self.r

            #Judging whether the system has arrived at the adjustment time
            for i4 in range(9):
                d_temp[i4] = d_temp[i4 + 1]
                phi_temp[i4] = phi_temp[i4 + 1]
            d_temp[9] = my_d
            phi_temp[9] = my_phi

            if tt >= 9:
                if self.flag_stop(d_temp, phi_temp) == 1:
                    temp_loss = (tt - 9) * dt
                    return temp_loss

        temp_loss = max_step * dt
        return temp_loss

if __name__ == '__main__':
    pi2 = RL_PI2()
    pi2.sample_using_PI2()


