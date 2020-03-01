import numpy as np
# from load import *
import math
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import multiprocessing as mp
import copy


'''------------------------------超参数模块------------------------------'''
# 辅助模块
linewidth = 1 # 绘图中曲线宽度
fontsize = 5 # 绘图字体大小
markersize = 2.5  # 标志中字体大小
legend_font_size = 5 #图例中字体大小

# 飞行器环境
PITCH_D = 1.0   # 俯仰角目标【度】

# PID模块
Overshoot_target = 0.02/57.3    # 超参数目标
ts_target = 400                 # 目标调整时间
Static_error_target = 0.02/57.3 # 静态误差
adjust_bound = 0.02             # 波动限制
belief_times = 50               # 稳定次数

Ki_Min = 0
Ki_Max = 100.0
Kp_Min = 0
Kp_Max = 100.0
Kd_Min = 0
Kd_Max = 100.0

# 强化学习
RAND_INIT = True
training_times = 100            # 训练次数
roll_outs = 20                  # 并行数量
Max_Adjust_times = 1000         # 最大调整次数

'''------------------------------辅助模块------------------------------'''
#按照path创建文件夹 e.g. mkdir("./figure/")
def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' Created successfully')
        return True
    else:
        print(path + ' Directory already exists')
        return False

# 存储实验数据 e.g. save_data("./data/","loss.txt",loss)
def save_data(dir,name,data):
    mkdir(dir)
    np.savetxt(dir+name,data)

# 读取实验数据 e.g. save_data("./data/loss.txt")
def read_data(dir):
    data = np.loadtxt(dir)
    return data

# 保存实验图片 e.g. save_figure("./figure/","loss.pdf")
def save_figure(dir,name):
    mkdir(dir)
    plt.savefig(dir+name,bbox_inches = 'tight')

# 绘制控制曲线
def plot_result(alpha_list,delta_z_list,theta_list,theta_desire_list,figure_number = 3):
    '''绘图参数定义'''
    label = ["Optimal Parameters using FR-PI2 ","Adjusted parameters ","Reference parameter","Reference parameter"]
    color = ["r","g","b","k"]
    line_style = ["-","-.",":","--"]
    '''绘制alpha曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.005s)$",fontproperties='Times New Roman',fontsize=fontsize)
    plt.ylabel("Attack Angle $(Degree)$",fontproperties='Times New Roman',fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(alpha_list[i], label=label[i],color=color[i],linestyle=line_style[i],linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp1/", "alpha_Curve.pdf")
    plt.show()
    '''绘制delta_z曲线'''
    plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman',fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman',fontsize=fontsize)
    plt.xlabel("Time$(0.005s)$",fontproperties='Times New Roman',fontsize=fontsize)
    plt.ylabel("Elevator $(Degree)$",fontproperties='Times New Roman',fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(delta_z_list[i], label=label[i], color=color[i], linestyle=line_style[i],linewidth=linewidth)
    plt.legend(loc='best',  prop={'family':'Times New Roman', 'size':legend_font_size})
    save_figure("./photo/exp1/", "delta_z_Curve.pdf")
    plt.show()
    '''绘制theta曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.005s)$",fontproperties='Times New Roman',fontsize=fontsize)
    plt.ylabel("theta Angle $(Degree)$",fontproperties='Times New Roman',fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(theta_list[i], label=label[i], color=color[i], linestyle=line_style[i],linewidth=linewidth)
    plt.plot(theta_desire_list[0], label="$\\theta_{target}$", linestyle="--",linewidth=linewidth)
    plt.legend(loc='best', prop={'family':'Times New Roman', 'size':legend_font_size})
    save_figure("./photo/exp1/", "theta_Curve.pdf")
    plt.show()

# 对比方法时绘制损失变化曲线
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
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    # plt.title("Loss Function Curve With Natural PI2", fontdict={'family': 'Times New Roman'})
    save_figure("./photo/exp1/", "loss.pdf")
    plt.show()


'''------------------------------动力学模型板块------------------------------'''

#------------------------------飞行器环境------------------------------
'''
飞行器的物理模型。
关键变量：
输出表现值：
    self.observation
目标参数：
    self.pitch_desired = PITCH_D / 57.3     # 俯仰角
    self.dpithch_desired = 0.0              # 俯仰角速度
    self.theta_desired = 0.0                # 飞行器角度
    self.dtheta_desired = 0.0               # 飞行器角速度
'''
class Planes_Env:
    # 初始化
    def __init__(self):
        # self.actions = [0,1]
        self.observation_dim = 2
        # 一阶变量
        self.altitude = 33500.0     # y高度【米】
        self.Mach = 15.0            # M速度【马赫】
        self.theta = 0.0            # 飞行角【弧度】
        self.pitch = 0.0 / 57.3     # 俯仰角【弧度】
        self.rrange = 0.0           # X距离【米】
        self.mass = 83191           # 质量【千克】
        self.omega_z = 0.0          # 俯仰角速度【弧度/秒】
        # 速度
        self.daltitude = 0.0            # 高度速度【米/秒】
        self.dMach = 0.0                # 速度变化【马赫/秒】
        self.dtheta = 0.0               # theta变化【弧度/秒】
        self.dpitch = 0.0               # 俯仰角速度【弧度/秒】
        self.drrange = 0.0              # 水平速度【米/秒】
        self.dmass = 0.0                # 质量变化【千克/秒】
        self.domega_z = 0.0             # 俯仰角加速度【弧度/秒2】
        # 攻角
        self.arfa = 0.0 / 57.3          # 攻角【弧度】
        # 目标
        self.pitch_desired = PITCH_D / 57.3     # 俯仰角目标【弧度】
        self.dpithch_desired = 0.0              # 俯仰角速度【弧度/秒】
        self.theta_desired = 0.0                # 飞行角目标【弧度】
        self.dtheta_desired = 0.0               # 飞行角速度【弧度/秒】
        #其他常量
        self.Vs = 305.58                # 换算单位【马赫/米】
        self.Lr =24.38                  # 纵向长度【米】
        self.G0 =9.81                   # 重力加速度【米/秒^2】
        self.Sr = 334.73                # 参考横截面积【米^2】
        self.Jz = 8288700               # Jz转动惯量
        self.Re = 6371000               # 地球半径【米】

        # 状态
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        self.max_steps = 400

        #角度阈值 攻角alpha[-1,10];舵偏delta_z[-20,20]
        self.alpha_threshold_max = 10       # 最大攻角【度】
        self.alpha_threshold_min = -1       # 最小攻角【度】
        self.delta_z_threhold_max = 20      # 最大舵偏【度】
        self.delta_z_threhold_min = -20     # 最小舵偏【度】

        # 时间
        self.tau = 0.005

    # 重设环境
    def reset(self):
        '''
        重设环境为初始状态
        :return: 新的观察值
        '''
        n = np.random.randint(1,1000,1)
        np.random.seed(n)
        self.observation = np.array([0.0, 0.0])
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
        # 重置目标
        self.pitch_desired = PITCH_D / 57.3
        self.dpithch_desired = 0.0
        self.theta_desired = 0.0
        self.dtheta_desired = 0.0
        self.steps_beyond_done = 0
        return self.observation

    # 采取动作
    def step(self, action):
        action = action[0]
        if action < -20:
            action = -20
        elif action > 20:
            action = 20
        # 变为度数
        Alpha_deg = self.arfa*57.3
        Jzc = 99.635
        # 空气密度rho
        Rho = np.exp(-2.114 * 10.0 ** (-14.0) * self.altitude ** 3.0 + 3.656 * 10.0 ** (-9.0) * self.altitude ** 2.0 - 3.309 * 10.0 ** (-4.0) * self.altitude + 3.217)
        # 动压q
        Qdyn = 0.5 * Rho * self.Mach * self.Vs * self.Mach * self.Vs
        # ***************************** 高超声速 	升力系数 ********************
        CL0 = - 8.19 * (10.0 ** (-2)) \
              + 4.70 * (10.0 ** (-2)) * self.Mach \
              + 1.86 * (10.0 ** (-2)) * Alpha_deg \
              - 4.73 * (10.0 ** (-4)) * (Alpha_deg * self.Mach) \
              - 9.19 * (10.0 ** (-3)) * (self.Mach ** 2) \
              - 1.52 * (10.0 ** (-4)) * (Alpha_deg ** 2) \
              + 7.74 * (10.0 ** (-4)) * (self.Mach ** 3) \
              + 4.08 * (10.0 ** (-6)) * (Alpha_deg ** 3) \
              + 5.99 * (10.0 ** (-7)) * ((Alpha_deg * self.Mach) ** 2) \
              - 2.93 * (10.0 ** (-5)) * (self.Mach ** 4) \
              - 3.91 * (10.0 ** (-7)) * (Alpha_deg ** 4) \
              + 4.12 * (10.0 ** (-7)) * (self.Mach ** 5) \
              + 1.30 * (10.0 ** (-8)) * (Alpha_deg ** 5)
        CL_e = - 1.45 * (10.0 ** (-5)) \
               + 7.10 * (10.0 ** (-6)) * self.Mach \
               + 1.01 * (10.0 ** (-4)) * Alpha_deg  \
               - 4.14 * (10.0 ** (-4)) * action \
               - 3.51 * (10.0 ** (-6)) * Alpha_deg * action \
               + 8.72 * (10.0 ** (-6)) * self.Mach * action \
               + 1.70 * (10.0 ** (-7)) * self.Mach * Alpha_deg * action
        CL_a = CL_e
        # ***************************** 高超声速 阻力系数 ********************
        CD0 =  8.7173 * (10.0 ** (-2)) \
              + 3.179 * (10.0 ** (-3)) * Alpha_deg \
              - 3.307 * (10.0 ** (-2)) * self.Mach  \
              - 1.250 * (10.0 ** (-4)) * (Alpha_deg * self.Mach) \
              + 5.036 * (10.0 ** (-3)) * (self.Mach ** 2) \
              - 1.100 * (10.0 ** (-3)) * (Alpha_deg ** 2)\
              + 1.405 * (10.0 ** (-7)) * ((Alpha_deg * self.Mach) ** 2) \
              - 3.658 * (10.0 ** (-4)) * (self.Mach ** 3) \
              + 3.175 * (10.0 ** (-4)) * (Alpha_deg ** 3) \
              + 1.274 * (10.0 ** (-5)) * (self.Mach ** 4) \
              - 2.985 * (10.0 ** (-5)) * (Alpha_deg ** 4) \
              - 1.705 * (10.0 ** (-7)) * (self.Mach ** 5) \
              + 9.766 * (10.0 ** (-7)) * (Alpha_deg ** 5)
        CD_e =   4.5548 * (10.0 ** (-4)) \
               - 1.1436 * (10.0 ** (-4)) * self.Mach \
               + 2.5411 * (10.0 ** (-5)) * Alpha_deg \
               - 3.6417 * (10.0 ** (-5)) * action \
               - 5.3015 * (10.0 ** (-7)) * self.Mach * Alpha_deg * action \
               + 3.0140 * (10.0 ** (-6)) * (self.Mach ** 2) \
               + 3.2187 * (10.0 ** (-6)) * (Alpha_deg ** 2)  \
               + 6.9629 * (10.0 ** (-6)) * (action ** 2)\
               + 2.1026 * (10.0 ** (-12)) * ((self.Mach * Alpha_deg * action) ** 2)
        CD_a = CD_e
        CD_r = 7.50 * (10.0 ** (-4)) \
               - 2.29 * (10.0 ** (-5)) * Alpha_deg \
               - 9.69 * (10.0 ** (-5)) * self.Mach \
               + 8.76 * (10.0 ** (-7)) * Alpha_deg ** 2 \
               + 2.70 * (10.0 ** (-6)) * self.Mach ** 2

        # 阻力系数
        CD = CD0 + CD_e + CD_a + CD_r
        # 升力系数
        CL = CL0 + CL_e + CL_a
        # ***************************** 高超声速 俯仰力矩 ********************
        mz0 = - 2.192 * (10.0 ** (-2)) \
              + 7.739 * (10.0 ** (-3)) * self.Mach \
              - 2.260 * (10.0 ** (-3)) * Alpha_deg \
              + 1.808 * (10.0 ** (-4)) * (Alpha_deg * self.Mach) \
              - 8.849 * (10.0 ** (-4)) * (self.Mach ** 2) \
              + 2.616 * (10.0 ** (-4)) * (Alpha_deg ** 2)\
              - 2.880 * (10.0 ** (-7)) * ((Alpha_deg * self.Mach) ** 2) \
              + 4.617 * (10.0 ** (-5)) * (self.Mach ** 3) \
              - 7.887 * (10.0 ** (-5)) * (Alpha_deg ** 3) \
              - 1.143 * (10.0 ** (-6)) * (self.Mach ** 4) \
              + 8.288 * (10.0 ** (-6)) * (Alpha_deg ** 4) \
              + 1.082 * (10.0 ** (-8)) * (self.Mach ** 5) \
              - 2.789 * (10.0 ** (-7)) * (Alpha_deg ** 5)
        mz_e = - 5.67 * (10.0 ** (-5)) \
               - 1.51 * (10.0 ** (-6)) * self.Mach \
               - 6.59 * (10.0 ** (-5)) * Alpha_deg \
               + 2.89 * (10.0 ** (-4)) * action \
               + 4.48 * (10.0 ** (-6)) * Alpha_deg * action \
               - 4.46 * (10.0 ** (-6)) * self.Mach * Alpha_deg \
               - 5.87 * (10.0 ** (-6)) * self.Mach * action \
               + 9.72 * (10.0 ** (-8)) * self.Mach * Alpha_deg * action
        mz_a = mz_e
        mz_r = - 2.79 * (10.0 ** (-5)) * Alpha_deg \
               - 5.89 * (10.0 ** (-8)) * (Alpha_deg ** 2) \
               + 1.58 * (10.0 ** (-3)) * (self.Mach ** 2) \
               + 6.42 * (10.0 ** (-8)) * (Alpha_deg ** 3) \
               - 6.69 * (10.0 ** (-4)) * (self.Mach ** 3) \
               - 2.10 * (10.0 ** (-8)) * (Alpha_deg ** 4) \
               + 1.05 * (10.0 ** (-4)) * (self.Mach ** 4) \
               + 3.14 * (10.0 ** (-9)) * (Alpha_deg ** 5) \
               - 7.74 * (10.0 ** (-6)) * (self.Mach ** 5) \
               - 2.18 * (10.0 ** (-10)) * (Alpha_deg ** 6) \
               + 2.70 * (10.0 ** (-7)) * (self.Mach ** 6) \
               + 5.74 * (10.0 ** (-12)) * (Alpha_deg ** 7) \
               - 3.58 * (10.0 ** (-9)) * (self.Mach ** 7)
        mzz = -1.36 \
              + 0.386 * self.Mach \
              + 7.85 * (10.0 ** (-4)) * Alpha_deg \
              + 1.40 * (10.0 ** (-4)) * Alpha_deg * self.Mach \
              - 5.42 * (10.0 ** (-2)) * (self.Mach ** 2) \
              + 2.36 * (10.0 ** (-3)) * (Alpha_deg ** 2) \
              - 1.95 * (10.0 ** (-6)) * ((Alpha_deg * self.Mach) ** 2) \
              + 3.80 * (10.0 ** (-3)) * (self.Mach ** 3) \
              - 1.48 * (10.0 ** (-3)) * (Alpha_deg ** 3) \
              - 1.30 * (10.0 ** (-4)) * (self.Mach ** 4)\
              + 1.69 * (10.0 ** (-4)) * (Alpha_deg ** 4) \
              + 1.71 * (10.0 ** (-6)) * (self.Mach ** 5)\
              - 5.93 * (10.0 ** (-6)) * (Alpha_deg ** 5)
        # 俯仰力矩系数
        mz = mz0 + mz_e + mz_a + mz_r + mzz * self.omega_z * 57.3 * self.Lr / (2 * self.Mach * self.Vs)
        # 升力
        Lift = Qdyn * CL * self.Sr
        # 阻力
        Drag = Qdyn * CD * self.Sr
        # 俯仰力矩
        Mz = Qdyn * mz * self.Sr * self.Lr
        # 推力 P
        Thrust = 1.9 * (10.0 ** 5.0)
        # 比冲Isp
        Isp = 4900
        #--------------------------动力学方程---------------------------
        # y速度【m/s】
        self.daltitude = self.Mach * self.Vs * np.sin(self.theta)
        # v变化【马赫/s】
        self.dMach = (Thrust * np.cos(self.arfa) - Drag - self.mass * self.G0 * np.sin(self.theta)) / (self.mass *self.Vs)
        # 飞行角速度
        self.dtheta = (Thrust * np.sin(self.arfa) + Lift) / (self.mass * self.Mach * self.Vs) + np.cos(self.theta) \
                      * (self.Mach * self.Vs / (self.Re + self.altitude) - self.G0 / (self.Mach * self.Vs))
        # 质量变化
        self.dmass = -Thrust / Isp
        # X变化
        self.drrange = self.Mach * self.Vs * np.cos(self.theta) * (self.Re / (self.Re + self.altitude))
        # omega_z的变化
        self.domega_z = Mz / self.Jz
        self.dpitch = self.omega_z
        # 更新
        self.Jz = self.Jz + Jzc * self.dmass * self.tau
        self.altitude = self.altitude + self.daltitude * self.tau
        self.Mach = self.Mach+self.dMach * self.tau
        self.theta = self.theta + self.dtheta * self.tau
        self.rrange = self.rrange + self.drrange * self.tau
        self.mass = self.mass + self.dmass * self.tau
        self.omega_z = self.omega_z + self.domega_z * self.tau
        self.pitch = self.pitch + self.dpitch * self.tau
        self.arfa = self.pitch - self.theta

        self.steps_beyond_done += 1

        self.observation = np.array([self.pitch*57.3, self.dpitch*57.3])
        return self.observation


'''------------------------------PID控制器板块------------------------------'''
# ------------------------------PID模型------------------------------
'''
PID模拟模块
主要功能：
    1 在env的基础上，复制环境进行模拟
        get_epsolid_reward(self,env, k1=1.5, k2=2.5, k3=0.5)
        返回：reward, done, ts, OS, SE
    2 对环境进行更新
        get_new_env(self, env,step_time ,k1=1.5, k2=2.5, k3=0.5)
        返回：env, alpha, dez_list, theta, desired_theta
    3 对环境进行模拟并且绘图
        model_simulation(self, k1=1.5, k2=2.5, k3=0.5, total_step=800)
        返回：alpha, dez_list, theta, desired_theta, error_list
'''
class PID_model():
    def __init__(self):
        self.env = Planes_Env()
    # 进行一次PID模拟
    def get_epsolid_reward(self,env, k1=1.5, k2=2.5, k3=0.5):
        '''
        进行PID模拟全程
        :param env: 环境
        :param k1: kp
        :param k2: kd
        :param k3: ki
        :return: 损失值
        '''
        total_step = 5000
        self.env = copy.deepcopy(env)
        alpha = []
        theta = []
        desired_theta = []
        q = []
        time = []
        control = []
        ierror = 0
        derror_list = []
        error_list = []
        dez_list = []
        tp = 0 # 峰值时间
        ts = total_step # 调整时间
        count = 0
        if_alpha = False
        # ----------进行循环控制----------
        for i in range(total_step):
            if count >= belief_times and ts == total_step:
                ts = i
                break
            error =  self.env.pitch_desired*57.3   - self.env.observation[0]
            derror = self.env.dpithch_desired*57.3 - self.env.observation[1]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror
            dez_list.append(action)
            # ----------系统分析----------
            # TODO:超调时间
            if (error == 0 and tp == 0):
                tp = i
            self.env.step(np.array([action]))
            alpha.append(self.env.arfa*57.3)
            if self.env.arfa*57.3 < -1 or self.env.arfa*57.3 >10:
                if_alpha = True
            theta.append(self.env.observation[0])
            desired_theta.append(self.env.pitch_desired*57.3)
            q.append(self.env.observation[1])
            time.append(i)
            control.append(action)

            if(abs(error)<=abs(adjust_bound * self.env.pitch)):
                count += 1
            else:
                count = 0
        # ----------系统分析----------
        # 超调
        Overshoot = max(np.array(theta)) - max(np.array(desired_theta))
        if Overshoot < Overshoot_target:
            Overshoot = 0
        else:
            Overshoot = (Overshoot - Overshoot_target)/Overshoot_target
        # 调整时间
        if ts <= ts_target:
            ts = 0
        else:
            ts = (ts - ts_target) / ts_target
        # 稳态误差
        st_error = 0.0
        for i in range(10):
            st_error += abs(error_list[-i])
        st_error /= 10.0
        if st_error < Static_error_target:
            Static_error = 0
        else:
            Static_error = (st_error-Static_error_target)/Static_error_target
        done = True
        # TODO：目前的奖励设置不合理
        reward = ts + Overshoot + Static_error
        if if_alpha:
            reward += 4500
            done = False
        return reward, done, ts, Overshoot, Static_error

    def get_new_env(self, env,step_time ,k1=1.5, k2=2.5, k3=0.5):
        '''
        对环境进行更新
        :param env: 原环境
        :param step_time: 更新步数
        :param k1: kp
        :param k2: kd
        :param k3: ki
        :return: step后的新的环境，alpha, dez_list, theta, desired_theta
        '''
        self.env = copy.deepcopy(env)
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
            # 结束更新
            if i == step_time:
                break
            # 进行PID控制
            error = (self.env.pitch_desired * 57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired * 57.3 - self.env.observation[1]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            # 在当前情况下进行进行控制
            action = np.clip(k1 * error + k2 * derror + k3 * ierror,-20,20)
            action_list.append(action)
            dez_list.append(action)
            self.env.step(np.array([action]))
            alpha.append(self.env.arfa * 57.3)
            theta.append(self.env.observation[0])
            desired_theta.append(self.env.pitch_desired * 57.3)
            q.append(self.env.observation[1])
            time.append(i)
            control.append(action)
            i = i + 1
        #TODO:必须把这个环境赋值给全局环境！
        return self.env,alpha, dez_list, theta, desired_theta

    def model_simulation(self,k1=1.5, k2=2.5, k3=0.5, total_step=800):
        '''
        按照当前参数，模拟total_step步，并且绘制theta图
        :param k1: kp
        :param k2: kd
        :param k3: ki
        :param total_step: 模拟步长
        :return: alpha, dez_list, theta, desired_theta
        '''
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
        while i < total_step:
            error = (self.env.pitch_desired * 57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired * 57.3 - self.env.observation[1]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            # action = k1 * error + k2 * derror + k3 * ierror
            action = np.clip(k1 * error + k2 * derror + k3 * ierror, -20, 20)
            # print("action:"+str(action)+"\t"+"error:"+str(error))

            dez_list.append(action)
            # 采取动作
            self.env.step(np.array([action]))
            alpha.append(self.env.arfa * 57.3)
            theta.append(self.env.observation[0])
            desired_theta.append(self.env.pitch_desired * 57.3)
            q.append(self.env.observation[1])
            time.append(i)
            control.append(action)
            i = i + 1
        plt.figure(2)
        plt.plot(time, theta, label="time-theta")
        plt.plot(time, desired_theta, label="time-desired_theta")
        plt.legend(loc="best")
        plt.title("This is the K epoch")
        plt.show()
        return alpha, dez_list, theta, desired_theta, error_list

    def model_simulation_env(self, env, k1=1.5, k2=2.5, k3=0.5, total_step=800):
        '''
        按照当前参数，模拟total_step步，并且绘制theta图
        :param k1: kp
        :param k2: kd
        :param k3: ki
        :param total_step: 模拟步长
        :return: alpha, dez_list, theta, desired_theta
        '''
        self.env = copy.deepcopy(env)
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
        while i < total_step:
            error = (self.env.pitch_desired * 57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired * 57.3 - self.env.observation[1]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            # action = k1 * error + k2 * derror + k3 * ierror
            action = np.clip(k1 * error + k2 * derror + k3 * ierror, -20, 20)
            # print("action:"+str(action)+"\t"+"error:"+str(error))

            dez_list.append(action)
            # 采取动作
            self.env.step(np.array([action]))
            alpha.append(self.env.arfa * 57.3)
            theta.append(self.env.observation[0])
            desired_theta.append(self.env.pitch_desired * 57.3)
            q.append(self.env.observation[1])
            time.append(i)
            control.append(action)
            i = i + 1
        plt.figure(2)
        plt.plot(time, theta, label="time-theta")
        plt.plot(time, desired_theta, label="time-desired_theta")
        plt.legend(loc="best")
        plt.title("This is the K epoch")
        plt.show()
        return alpha, dez_list, theta, desired_theta, error_list

'''------------------------------FR-PI2模块------------------------------'''


# 归一化函数
def ZscoreNormalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
"""[0,1] normaliaztion"""
def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


""" -----------------------------------------------------------强化学习部分-------------------------------------------------------------"""
# 训练次数,也就是策略迭代次数

class RL_PI2:

    # 初始化
    def __init__(self,if_filter=RAND_INIT, attenuation_step_length=10, alpha=0.85):
        ## 滚动优化,用于记录当前阶段
        self.env = Planes_Env()
        # 最佳K参数
        self.K_opl = np.zeros((3,1))
        # 最优损失
        self.Loss_opl = 1e10
        # 初始化的Loss
        self.Loss_end = np.array([1e10,1e10,1e10])
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
        self.loss = np.zeros((roll_outs, 5), dtype=np.float64)
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
        self.random_init = True
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

    # 记录数据
    def data_record(self):
        np.savetxt('./data/loss_after_training.txt',self.loss_after_training)
        np.savetxt('./data/K_after_training.txt',self.K_after_training)

    # PI2模块——使用滚动优化学习
    def sample_using_PI2(self):
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

    # 初始化数值
    def set_initial_value(self,INIT_K=[1.5,2.5,0.5]):
        if self.random_init:
            self.K[0] = random.uniform(Kp_Min,Kp_Max)
            self.K[1] = random.uniform(Kd_Min,Kd_Max)
            self.K[2] = random.uniform(Ki_Min,Ki_Max)
            self.K_opl = self.K
        else:
            self.K0 = INIT_K
            self.K[0] = INIT_K[0]
            self.K[1] = INIT_K[1]
            self.K[2] = INIT_K[2]
            self.K_opl = self.K
        # 初始化方差
        self.sigma[0] = 1.0
        self.sigma[1] = 0.3
        self.sigma[2] = 0.1
        # 初始化记录参数
        self.current_roll = 0
        self.current_training = 0
        # 初始化滚动环境
        self.env.reset()

    # 计算第j条轨迹的损失值，得到delta1,delta2,delta3；cur_k1,cur_k2,cur_k3；loss
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
        '''
        if self.if_filter and (self.current_training > 1 and loss[0] > self.loss_after_training[self.current_training - 1]):
            delta1 = delta2 = delta3 = 0.0
            cur_k1 = self.K[0] + delta1
            cur_k2 = self.K[1] + delta2
            cur_k3 = self.K[2] + delta3
        '''
        return delta1,delta2,delta3,cur_k1,cur_k2,cur_k3,loss

    # 策略评估，对N调轨迹并行，并且更新self.K_delta和self.K_roll和loss
    def policy_evl(self):
        multi_res = [poll.apply_async(self.cal_trajectory_loss, (j,)) for j in range(roll_outs)]
        for j, res in enumerate(multi_res):
            self.k_delta[0, j] = res.get()[0]
            self.k_delta[1, j] = res.get()[1]
            self.k_delta[2, j] = res.get()[2]
            self.K_roll[0, j] = res.get()[3]
            self.K_roll[1, j] = res.get()[4]
            self.K_roll[2, j] = res.get()[5]
            self.loss[j] = res.get()[6]
    # 策略改善，对当前K进行路径积分改进
    # NOTE：当前使用reward拆分作为方差更新的依据
    def policy_improve(self):
        exponential_value_loss = np.zeros((roll_outs, 1), dtype=np.float64)
        probability_weighting = np.zeros((roll_outs, 1), dtype=np.float64)
        loss = np.array(self.loss).T
        # --------------------更新方向--------------------
        # 现在loss 是5*20—— loss, done, ts, Overshoot, Static_error
        # TODO: 此部分为使用reward拆分作为方向
        '''
        # P——ts
        type_sigma = np.array([0,0,0],np.bool)
        TYPE_LOSS = 2
        if loss[TYPE_LOSS].max() == loss[TYPE_LOSS].min():
            for i2 in range(roll_outs):
                probability_weighting[i2] = 1/roll_outs
        else:
            for i2 in range(roll_outs):
                exponential_value_loss[i2] = np.exp(-self.PI2_coefficient * (loss[TYPE_LOSS, i2] - loss[TYPE_LOSS].min())
                                                    / (loss[TYPE_LOSS].max() - loss[TYPE_LOSS].min()))
            for i2 in range(roll_outs):
                probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)
        temp_k = np.dot(self.k_delta[0], probability_weighting)
        # self.sigma[0] = loss[TYPE_LOSS].min()
        type_sigma[0] = (self.Loss_end[0] < loss[TYPE_LOSS].min())
        self.K[0] = self.K[0] + temp_k[0]
        # PID方差更改——loss——

        ## D——OS
        TYPE_LOSS = 3
        if loss[TYPE_LOSS].max() == loss[TYPE_LOSS].min():
            for i2 in range(roll_outs):
                probability_weighting[i2] = 1 / roll_outs
        else:
            for i2 in range(roll_outs):
                exponential_value_loss[i2] = np.exp(
                    -self.PI2_coefficient * (loss[TYPE_LOSS, i2] - loss[TYPE_LOSS].min())
                    / (loss[TYPE_LOSS].max() - loss[TYPE_LOSS].min()))
            for i2 in range(roll_outs):
                probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)
        temp_k = np.dot(self.k_delta[1], probability_weighting)
        type_sigma[1] = (self.Loss_end[1] < loss[TYPE_LOSS].min())
        # self.sigma[1] = loss[TYPE_LOSS].min()
        self.K[1] = self.K[1] + temp_k

        ## I——E
        TYPE_LOSS = 4
        if loss[TYPE_LOSS].max() == loss[TYPE_LOSS].min():
            for i2 in range(roll_outs):
                probability_weighting[i2] = 1 / roll_outs
        else:
            for i2 in range(roll_outs):
                exponential_value_loss[i2] = np.exp(
                    -self.PI2_coefficient * (loss[TYPE_LOSS, i2] - loss[TYPE_LOSS].min())
                    / (loss[TYPE_LOSS].max() - loss[TYPE_LOSS].min()))
            for i2 in range(roll_outs):
                probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)
        temp_k = np.dot(self.k_delta[2], probability_weighting)
        type_sigma[2] = (self.Loss_end[2] < loss[TYPE_LOSS].min())
        # self.sigma[2] = loss[TYPE_LOSS].min()
        self.K[2] = self.K[2] + temp_k
        self.Loss_end = np.array([loss[2].min(), loss[3].min(), loss[4].min()])
        '''
        # TODO：此部分为使用reward作为方向
        for i2 in range(roll_outs):
            exponential_value_loss[i2] = np.exp(-self.PI2_coefficient * (loss[0, i2] - loss[0].min())
                                                / (loss[0].max() - loss[0].min()))
        for i2 in range(roll_outs):
            probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)
        temp_k = np.dot(self.k_delta, probability_weighting)
        reward, done, ts, Overshoot, Static_error = self.reward_model.get_epsolid_reward(self.env, self.K[0],self.K[1],self.K[2])
        # 确保K_opl的最优性
        best_K_ind = np.argmin(loss[0])
        if loss[0, best_K_ind] < self.Loss_opl:
            self.K_opl[0] = self.K_roll[0, best_K_ind]
            self.K_opl[1] = self.K_roll[1, best_K_ind]
            self.K_opl[2] = self.K_roll[2, best_K_ind]
            self.Loss_opl = loss[0, best_K_ind]
        if reward < self.Loss_opl:
            self.K_opl = self.K
            self.Loss_opl = reward
        #--------------------更新步长--------------------
        type_sigma = np.array([0, 0, 0], np.bool)
        type_sigma[0] = (ts <= 1e-5)
        type_sigma[1] = (Overshoot <= 1e-5)
        type_sigma[2] = (Static_error <= 1e-5)
        self.Loss_end = np.array([ts, Overshoot, Static_error])
        for id_ in range(3):
            if type_sigma[id_] == True:    # 说明原来的损失小
                self.sigma[id_] *= self.alpha
            else:
                self.sigma[id_] /= self.alpha
        print("当前的搜索方差")
        print(self.sigma)

        # --------------------更新K--------------------
        self.K = self.K + temp_k

    # 判断迭代停止
    def iterator_finished(self):
        flag1 = sum((self.K_after_training[:, self.current_training - 1] - self.K_after_training[:, self.current_training]) ** 2) <= 1e-6
        flag2 = self.loss_after_training[self.current_training]
        if flag1 < 1e-6 and flag2 < 0.5:
            return True
        else:
            return False
    # 强化学习N次，输出self.K；self.K_after_training；self.loss_after_training；self.current_training+1
    def training(self):
        theta_data_in_rolling = []
        i = 0
        self.K_after_training[:, self.current_training] = self.K[:, 0]
        self.loss_after_training[self.current_training] = self.reward_model.get_epsolid_reward(self.env, self.K[0],
                                                                                               self.K[1],self.K[2])[0]
        self.Loss_opl = 1e10
        # 初始化新环境
        env_new = self.env
        while i < training_times:
            i += 1
            self.current_training = i
            # --------------------模拟仿真--------------------
            if self.current_training % self.attenuation_step_length == 0 and self.current_training != 0:
                if self.current_training %3 == 0:
                    # 绘制损失变化
                    plt.plot(self.loss_after_training[self.current_training - 3:self.current_training])
                    plt.title("loss between %d and %d epoch"%(self.current_training - 3,self.current_training))
                    plt.show()
                    print("当前的损失"+str(self.current_training))
                    # 使用新的环境进行情况模拟
                    self.reward_model.model_simulation_env(env_new, self.K[0], self.K[1], self.K[2], 10000)
                    # 进入下一个环境和新的目标——目前设定为15轮
                    if self.current_training %15 == 0:
                        env_new, alpha_, dez_list_, theta_, desired_theta_ = self.reward_model.get_new_env(env_new, ts_target, self.K[0], self.K[1], self.K[2])
                        # 为计算误差更新Env
                        self.env = env_new
                        for theta in theta_:
                            theta_data_in_rolling.append(theta)
                        self.reward_model.env.pitch_desired  += 1.0/57.3     # 每次目标增长1°
                        # 每15轮绘制当前控制曲线
                        plt.title("The stage update of the environment in time " + str(self.current_training))
                        plt.plot(theta_data_in_rolling)
                        plt.show()
                    if self.reward_model.env.pitch_desired > 5.0/57.3:
                        print("Finish!")
                    print("FUCK!!!!!!!!!!!!!!!!!")
                    print(self.env.pitch_desired*57.3)
                    print("FUCK!!!!!!!!!!!!!!!!!")
                    # 判断是否还原K
                    if self.Loss_opl < self.loss_after_training[self.current_training]:
                        self.K = self.K_opl
                        self.loss_after_training[self.current_training] = self.Loss_opl

            # --------------------策略评估和改进--------------------
            self.policy_evl()
            self.policy_improve()
            # 记录参数
            self.K_after_training[:, self.current_training] = self.K[:, 0]
            self.loss_after_training[self.current_training] = self.reward_model.get_epsolid_reward(self.env,self.K[0], self.K[1],self.K[2])[0]
            # 判断是否可以停止
            if self.iterator_finished():
                if self.loss_after_training[self.current_training] <= self.Loss_opl:
                    self.K_opl = self.K
                    self.Loss_opl = self.loss_after_training[self.current_training]
                break
            # 输出当前训练次数
            if(self.current_training % self.attenuation_step_length == 0 ):
                print(self.current_training,time.time()-first_time)
        # 总共有training_times+1 条数据
        for it in range(self.current_training,training_times+1):
            self.loss_after_training[it] = self.loss_after_training[self.current_training]
        # 重设最优
        if self.loss_after_training[self.current_training] > self.Loss_opl:
            self.K = self.K_opl
        return self.K[0],self.K[1],self.K[2],self.K_after_training,self.loss_after_training,self.current_training+1

    # 滚动优化
    def rolling_optimization(self,rolling_time=20,total_step=200):
        # 优化次数
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
        while iterator < opt_times:
            # 每次策略迭代之后的损失函数    动态变量 每一次策略迭代刷新一次
            self.loss_after_training = np.zeros((training_times + Max_Adjust_times, 1), dtype=np.float64)
            # 每次策略迭代之后的K  动态变量 每一次策略迭代刷新一次
            self.K_after_training = np.zeros((3, training_times + Max_Adjust_times), dtype=np.float64)
            self.current_training = 0
            self.training()
            max_time = max_time+ self.current_training
            ## 迭代多少次和当前训练没有关系
            self.FR_time[iterator] = self.current_training
            for i in range(self.current_training+1):
                K_list.append(self.K_after_training[:,i])
                loss_list.append(self.loss_after_training[i])
            plt.plot(self.loss_after_training)
            plt.title("1111111Time1111")
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
        plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
        # 图上的legend，记住字体是要用prop以字典形式设置的，而且字的大小是size不是fontsize，这个容易和xticks的命令弄混
        # plt.title("$\mathcal{K}^{*}$ Iteration Graph",fontdict={'family': 'Times New Roman'})
        save_figure("./photo/exp1/",
                    "time.pdf" )
        plt.show()

        return alpha_list,delta_z_list,theta_list,theta_desired_list,K_list,loss_list,max_time+1


'''------------------------------主函数------------------------------'''
if __name__ == "__main__":
    print("START!")
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

    training_times = 100
    # model = RL_PI2(if_filter=True, attenuation_step_length=1, alpha=0.85)
    # model.set_initial_value([0, 0, 0])
    # alpha, delta_z, theta, theta_desired, k_after, loss, cur_time = model.rolling_optimization(rolling_time=20,
    #                                                                                            total_step=200)
    # alpha_list.append(alpha)
    # delta_z_list.append(delta_z)
    # theta_list.append(theta)
    # theta_desire_list.append(theta_desired)
    # k_list.append(k_after)
    # loss_list.append(loss)
    # maxtime = max(maxtime, cur_time)
    # training_times = maxtime
    # # ## 第1组优化10次
    model = RL_PI2(if_filter=True,attenuation_step_length=1,alpha=0.85)
    model.set_initial_value([0, 0, 0])
    k[0],k[1],k[2],k_after,loss,cur_time = model.training()
    alpha, delta_z, theta, theta_desired ,elist= reward_model.model_simulation(k[0],k[1],k[2],1000)
    alpha_list.append(alpha)
    delta_z_list.append(delta_z)
    theta_list.append(theta)
    theta_desire_list.append(theta_desired)
    k_list.append(k_after)
    loss_list.append(loss)
    training_times = maxtime
    # # ## 第2组优化10次
    # model = RL_PI2(if_filter=False,attenuation_step_length=5,alpha=1/0.85)
    # model.set_initial_value([100, 100, 100])
    # k[0], k[1], k[2], k_after, loss, cur_time = model.training()
    # alpha, delta_z, theta, theta_desired = reward_model.model_simulation(k[0], k[1], k[2],200)
    # alpha_list.append(alpha)
    # delta_z_list.append(delta_z)
    # theta_list.append(theta)
    # theta_desire_list.append(theta_desired)
    # k_list.append(k_after)
    # loss_list.append(loss)
    # maxtime = max(maxtime, cur_time)
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
    plot_result(alpha_list,delta_z_list,theta_list,theta_desire_list,figure_number=1)
    plot_loss_k(k_list, loss_list, maxtime,figure_number=1)
