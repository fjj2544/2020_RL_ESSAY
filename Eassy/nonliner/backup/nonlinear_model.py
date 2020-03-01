

'''------------------------------------------------------------------------------辅助板块--------------------------'''
'''导入相关函数库'''
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
'''全局参数定义'''
linewidth = 1 # 绘图中曲线宽度
fontsize = 5 # 绘图字体大小
markersize = 2.5  # 标志中字体大小
legend_font_size = 5 #图例中字体大小
'''辅助函数定义'''
'''
按照path创建文件夹 e.g. mkdir("./figure/")
'''
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
'''
存储实验数据 e.g. save_data("./data/","loss.txt",loss)
'''
def save_data(dir,name,data):
    mkdir(dir)
    np.savetxt(dir+name,data)
'''
读取实验数据 e.g. save_data("./data/loss.txt")
'''
def read_data(dir):
    data = np.loadtxt(dir)
    return data
'''
读取实验图片 e.g. save_figure("./figure/","loss.pdf")
'''
def save_figure(dir,name):
    mkdir(dir)
    plt.savefig(dir+name,bbox_inches = 'tight')
'''
绘制控制曲线
'''
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
'''------------------------------------------------------------------------------动力学模型板块--------------------------'''
# 高超声速飞行器定点飞行，高度33.5km，速度15Ma
#定义环境
class Planes_Env:
    def __init__(self):
        # self.actions = [0,1]
        self.observation_dim = 2
        # 一阶变量
        self.altitude = 33500.0     # y高度【米】
        self.Mach = 15.0            # M速度【马赫】
        self.theta = 0.0            # theta速度倾角【弧度】
        self.pitch = 0.0 / 57.3     # 俯仰角【弧度】
        # self.rrange = 0.0         # x距离【米】
        self.mass = 83191           # m质量【千克】
        self.omega_z = 0.0          # omega_z俯仰角速度【弧度/秒】
        self.Jz = 8288700           # Jz转动惯量【千克*米2】
        # 速度
        self.daltitude = 0.0            # Y速度【米/秒】
        self.dMach = 0.0                # 速度变化【马赫/秒】
        self.dtheta = 0.0               # theta变化【弧度/秒】
        self.dpitch = 0.0               # 俯仰角速度【弧度/秒】
        # self.drrange = 0.0            # X速度【米/秒】
        self.dmass = 0.0                # 质量变化速度【千克/s】
        self.domega_z = 0.0             # 俯仰角加速度【弧度/秒2】
        self.dJzc = 0.0                 # Jz转动惯量变化速度【千克*米2/秒】
        # 攻角
        self.arfa = 0.0 / 57.3          # 攻角【弧度】
        # 目标
        self.pitch_desired = 5 / 57.3   # 俯仰角【弧度】
        self.dpithch_desired = 0.0      # 俯仰角速度【弧度/秒】
        self.theta_desired = 0.0        # theta【弧度】
        self.dtheta_desired = 0.0       # theta速度【弧度/秒】
        #其他常量
        self.Vs = 305.58                # Vs速度换算【马赫/米】
        self.Lr =24.38                  # Lc纵向长度【米】
        self.G0 =9.81                   # g重力加速度【米/秒^2】
        self.Sr = 334.73                # S参考横截面积【米^2】
        self.Re = 6371000               # 地球半径【米】

        #状态量分别为[飞机迎角 alpha, 飞机俯仰角theta, 飞机俯仰角速度q]
        # self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.observation = np.array([0.0, 0.0])
        self.state = np.array([0.0,0.0])
        self.steps_beyond_done = 0
        self.max_steps = 400
        self.viewer = None

        #角度阈值 攻角alpha[-1,10];舵偏delta_z[-20,20]
        self.alpha_threshold_max = 10       # 最大攻角【度】
        self.alpha_threshold_min = -1       # 最小攻角【度】
        self.delta_z_threhold_max = 20      # 最大舵偏【度】
        self.delta_z_threhold_min = -20     # 最小舵偏【度】
        self.reward = 0.0
        self.cost = 0.0
        self.tau = 0.005

    def reset(self):
        '''
        重设环境为初始状态
        :return: 新的观察值
        '''
        n = np.random.randint(1,1000,1)
        np.random.seed(n)
        # 一阶变量
        self.altitude = 33500.0     # y高度【米】
        self.Mach = 15.0            # M速度【马赫】
        self.theta = 0.0            # theta速度倾角【弧度】
        self.pitch = 0.0 / 57.3     # 俯仰角【弧度】
        # self.rrange = 0.0         # x距离【米】
        self.mass = 83191           # m质量【千克】
        self.omega_z = 0.0          # omega_z俯仰角速度【弧度/秒】
        self.Jz = 8288700           # Jz转动惯量【千克*米2】
        # 速度
        self.daltitude = 0.0            # Y速度【米/秒】
        self.dMach = 0.0                # 速度变化【马赫/秒】
        self.dtheta = 0.0               # theta变化【弧度/秒】
        self.dpitch = 0.0               # 俯仰角速度【弧度/秒】
        # self.drrange = 0.0            # X速度【米/秒】
        self.dmass = 0.0                # 质量变化速度【千克/s】
        self.domega_z = 0.0             # 俯仰角加速度【弧度/秒2】
        self.dJzc = 0.0                 # Jz转动惯量变化速度【千克*米2/秒】
        # 攻角
        self.arfa = 0.0 / 57.3          # 攻角【弧度】
        # 目标
        self.pitch_desired = 5 / 57.3   # 俯仰角【弧度】
        self.dpithch_desired = 0.0      # 俯仰角速度【弧度/秒】
        self.theta_desired = 0.0        # theta【弧度】
        self.dtheta_desired = 0.0       # theta速度【弧度/秒】
        # 状态
        self.state = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        self.reward = 0.0
        self.cost = 0.0
        self.observation = np.array([0.0, 0.0])
        return self.observation

    def step(self,action):
        # action = np.clip(action[0],-20,20)
        action =action[0]
        # 变为度数
        Alpha_deg = self.arfa*57.3
        # 空气密度rho
        Rho = np.exp( - 2.114 * (10.0 ** (-14.0)) *(self.altitude ** 3.0)
                      + 3.656 * (10.0 ** (-9.0)) * (self.altitude ** 2.0)
                      - 3.309 * (10.0 ** (-4.0)) * self.altitude
                      + 3.217
                    )
        # 动压q
        Qdyn = 0.5 * Rho * self.Mach * self.Vs * self.Mach * self.Vs
        # ***************************** 高超声速 	升力系数 ********************
        # C_y0
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
        # C_y_theta_e
        CL_e = - 1.45 * (10.0 ** (-5)) \
               + 7.10 * (10.0 ** (-6)) * self.Mach \
               + 1.01 * (10.0 ** (-4)) * Alpha_deg  \
               - 4.14 * (10.0 ** (-4)) * action \
               - 3.51 * (10.0 ** (-6)) * Alpha_deg * action \
               + 8.72 * (10.0 ** (-6)) * self.Mach * action \
               + 1.70 * (10.0 ** (-7)) * self.Mach * Alpha_deg * action
        CL_a = CL_e
        # ***************************** 高超声速 阻力系数 ********************
        # C_x0
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
        # C_x_theta_e
        CD_e =   4.5548 * (10.0 ** (-4)) \
               - 1.1436 * (10.0 ** (-4)) * self.Mach \
               + 2.5411 * (10.0 ** (-5)) * Alpha_deg \
               - 3.6417 * (10.0 ** (-5)) * action \
               - 5.3015 * (10.0 ** (-7)) * self.Mach * Alpha_deg * action \
               + 3.0140 * (10.0 ** (-6)) * (self.Mach ** 2) \
               + 3.2187 * (10.0 ** (-6)) * (Alpha_deg ** 2)  \
               + 6.9629 * (10.0 ** (-6)) * (action ** 2)\
               + 2.1026 * (10.0 ** (-12)) * ((self.Mach * Alpha_deg * action) ** 2)
        # 未知含义
        CD_a = CD_e
        # TODO:未知公式
        CD_r = 7.50 * (10.0 ** (-4)) \
               - 2.29 * (10.0 ** (-5)) * Alpha_deg \
               - 9.69 * (10.0 ** (-5)) * self.Mach \
               + 8.76 * (10.0 ** (-7)) * Alpha_deg ** 2 \
               + 2.70 * (10.0 ** (-6)) * self.Mach ** 2
        # 阻力系数
        # TODO：多了CD_r
        CD = CD0 + CD_e + CD_a + CD_r
        # 升力系数
        CL = CL0 + CL_e + CL_a
        # ***************************** 高超声速 俯仰力矩 ********************
        # M_z0
        mz0 = - 2.192 * (10.0 ** (-2)) \
              + 7.739 * (10.0 ** (-3)) * self.Mach \
              - 2.260 * (10.0 ** (-3)) * Alpha_deg \
              + 1.808 * (10.0 ** (-4)) * (Alpha_deg * self.Mach) \
              + 8.849 * (10.0 ** (-4)) * (self.Mach ** 2) \
              + 2.616 * (10.0 ** (-4)) * (Alpha_deg ** 2)\
              - 2.880 * (10.0 ** (-7)) * ((Alpha_deg * self.Mach) ** 2) \
              + 4.617 * (10.0 ** (-5)) * (self.Mach ** 3) \
              - 7.887 * (10.0 ** (-5)) * (Alpha_deg ** 3) \
              - 1.143 * (10.0 ** (-6)) * (self.Mach ** 4) \
              + 8.288 * (10.0 ** (-6)) * (Alpha_deg ** 4) \
              + 1.082 * (10.0 ** (-8)) * (self.Mach ** 5) \
              - 2.789 * (10.0 ** (-7)) * (Alpha_deg ** 5)
        # M_z_theta_e
        mz_e = - 5.67 * (10.0 ** (-5)) \
               - 1.51 * (10.0 ** (-6)) * self.Mach \
               - 6.59 * (10.0 ** (-5)) * Alpha_deg \
               + 2.89 * (10.0 ** (-4)) * action \
               + 4.48 * (10.0 ** (-6)) * Alpha_deg * action \
               - 4.46 * (10.0 ** (-6)) * self.Mach * Alpha_deg \
               - 5.87 * (10.0 ** (-6)) * self.Mach * action \
               + 9.72 * (10.0 ** (-8)) * self.Mach * Alpha_deg * action
        # 未知定义
        mz_a = mz_e
        # TODO：未知公式
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
        # M_zz
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
        # TODO：此处omega_z应该变为角度制
        mz = mz0 + mz_e + mz_a + mz_r + mzz * 57.3 * self.omega_z * self.Lr / (2 * self.Mach * self.Vs)
        # 升力
        Lift = Qdyn * CL * self.Sr
        # 阻力
        Drag = Qdyn * CD * self.Sr
        # 俯仰力矩
        Mz = Qdyn * mz * self.Sr * self.Lr
        # 推力 P 【N】
        Thrust = 1.9 * (10.0 ** 5.0)
        # 比冲Isp【N*s/kg】
        Isp = 4900


        # 动力学方程
        #-----------------------------------------------------
        # y速度【m/s】 T
        self.daltitude = self.Mach * self.Vs * np.sin(self.theta)
        # x速度【m/s】 T
        # self.drrange = self.Mach * self.Vs * np.cos(self.theta) * (self.Re / (self.Re + self.altitude))
        # v变化【马赫/s2】 T
        self.dMach = (Thrust * np.cos(self.arfa) - Drag - self.mass * self.G0 * np.sin(self.theta)) / (self.mass * self.Vs)
        # v给的不一样——mgsin/m
        # theta角速度【?/s2】 T
        self.dtheta = (Thrust * np.sin(self.arfa) + Lift) / (self.mass * self.Mach * self.Vs) \
            + np.cos(self.theta) * (self.Mach * self.Vs / (self.Re + self.altitude) - self.G0 / (self.Mach * self.Vs))
        # 质量变化【kg/s】 N/A
        self.dmass = -Thrust / Isp
        # 转动惯量变化    N/A
        self.dJzc = -99.635 * self.dmass
        # omega_z的角速度 F
        # TODO：与孙老师给的冲突，检验原论文为孙老师的是对的
        self.domega_z = Mz / self.Jz
        # 俯仰角速度 T
        self.dpitch = self.omega_z

        # 变化
        self.Jz         = self.Jz       + self.dJzc * self.tau
        self.altitude   = self.altitude + self.daltitude * self.tau
        self.Mach       = self.Mach     + self.dMach * self.tau
        self.theta      = self.theta    + self.dtheta * self.tau
        # self.rrange     = self.rrange   + self.drrange * self.tau
        self.mass       = self.mass     + self.dmass * self.tau
        self.omega_z    = self.omega_z  + self.domega_z * self.tau
        self.pitch      = self.pitch    + self.dpitch * self.tau
        self.arfa       = np.clip(self.pitch - self.theta , -1/57.3,10/57.3)
        self.steps_beyond_done += 1
        #根据更新的状态判断是否结束
        lose = Alpha_deg < self.alpha_threshold_min or Alpha_deg > self.alpha_threshold_max
        #设置回报
        if not lose :
            self.reward =-((self.pitch*57.3-self.pitch_desired*57.3)**2+0.1*(self.dpitch*57.3-self.dpitch*57.3)**2+0.01*action**2)
        else:
            self.reward = -4500
        done = lose or self.steps_beyond_done > self.max_steps
        self.observation = np.array([self.pitch*57.3, self.dpitch*57.3])
        return self.observation, self.reward, done
'''------------------------------------------------------------PID控制器板块------------------------------------'''
'''
算法参数
'''
Overshoot_target = 1e-3
ts_target = 600
Waveform_oscillation_bound = 1
Static_error_target = 0.01
adjust_bound = 0.02
belief_times = 50
'''
PID控制器
'''
class PID_model():
    def __init__(self):
        self.env = Planes_Env()
    def get_epsolid_reward(self,env, k1=1.5, k2=2.5, k3=0.5):
        total_step = 10000
        self.env = copy.copy(env)
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
        tp = 0 # 峰值时间
        ts = total_step # 调整时间
        count = 0
        for i in range(total_step):
            if count >= belief_times and ts == total_step:
                ts = i
                break
            error = (self.env.pitch_desired*57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired*57.3-self.env.observation[1]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror
            dez_list.append(action)
            if (error == 0 and tp == 0):
                tp = i
            self.env.step(np.array([action]))
            alpha.append(self.env.arfa*57.3)
            theta.append(self.env.observation[0])
            desired_theta.append(self.env.pitch_desired*57.3)
            q.append(self.env.observation[1])
            time.append(i)
            control.append(action)
            if(abs(error)<=abs(adjust_bound * self.env.pitch)):
                count += 1
            else:
                count = 0
        '''超调量'''
        Overshoot = max(np.array(theta)) - max(np.array(desired_theta))
        Overshoot = 0 if Overshoot < Overshoot_target else (Overshoot - Overshoot_target) / Overshoot_target
        '''调整时间'''
        ts = 0 if ts <= ts_target else (ts - ts_target) / ts_target
        '''稳态误差'''
        Static_error = 0 if abs(error_list[-1]) < Static_error_target else abs(error_list[-1])
        r = Overshoot + Static_error
        return r
    def get_new_env(self, env,step_time ,k1=1.5, k2=2.5, k3=0.5):
        self.env = copy.copy(env)
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
            error = (self.env.pitch_desired * 57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired * 57.3 - self.env.observation[1]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror
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
        return self.env,alpha, dez_list, theta, desired_theta
    def model_simulation( self,k1=1.5, k2=2.5, k3=0.5, total_step=800):
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
            error = (self.env.pitch_desired * 57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired * 57.3 - self.env.observation[1]
            error_list.append(error)
            derror_list.append(derror)
            ierror = ierror + error * self.env.tau
            action = k1 * error + k2 * derror + k3 * ierror
            dez_list.append(action)
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
        return alpha, dez_list, theta, desired_theta
""" -----------------------------------------------------------FR-PI2板块-------------------------------------------------------------"""
'''归一化部分'''
"""Z-score normaliaztion"""
"""这种方法要求原始数据的分布可以近似为高斯分布，否则效果会很差。标准化公式如下 """
def ZscoreNormalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
"""[0,1] normaliaztion"""
def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
training_times = 100  # 训练次数
roll_outs = 20  # 路径数量
class RL_PI2:
    def __init__(self,IF_FILTER=True):
        self.env = Planes_Env() # 滚动优化,用于记录当前阶段
        self.K = np.zeros((3, 1), dtype=np.float64) # 记录每次策略迭代之后的K(包括初始化)
        self.K_roll = np.zeros((3, roll_outs), dtype=np.float64) # 记录每 roll_outs 局势内的 K
        self.K_record = np.zeros((3, roll_outs, training_times), dtype=np.float64) # 记录策略迭代中全部的K
        self.sigma = np.zeros((3, 1), dtype=np.float64) # 噪声方差,利用不同的方差控制不同参数的调节率, 从而实现分阶段控制
        self.k_delta = np.zeros((3, roll_outs), dtype=np.float64) # K变化量,用于存储噪声大小,进行策略改进
        self.loss = np.zeros((roll_outs, 1), dtype=np.float64) # 记录回报,用于策略改进
        self.loss_record = np.zeros((roll_outs, training_times), dtype=np.float64) # 记录所有策略迭代中损失函数
        self.loss_after_training = np.zeros((training_times, 1), dtype=np.float64) # 每次策略迭代之后的损失函数
        self.K_after_training = np.zeros((3, training_times), dtype=np.float64) # 每次策略迭代之后的K
        self.K_after_roll_step = np.zeros((3,2000),dtype=np.float64) # 每次滚动优化的K
        """ -----------------------------------------------------------定义算法超参数-------------------------------------------------------------"""
        self.attenuation_step_length = 1 # 方差更新间隔频率
        self.alpha = 0.85 # 方差更新系数
        self.current_roll = 0 # 记录当前第条轨迹
        self.current_training = 0  # 记录当前第几次策略迭代
        self.PI2_coefficient = 30.0  # PI2超参数 lambda
        self.reward_model = PID_model()         ## 交互模型
        self.save_data = False         ## 是否记录数据
        self.plot_photo = False         ## 是否绘图
        self.save_photo = True         ## 是否记录图片
        self.if_filter = IF_FILTER     # 是否滤波
    def data_record(self):
        save_data('./data/','loss_after_training.txt',self.loss_after_training)
        save_data('./data/','K_after_training.txt',self.K_after_training)
    def train_using_PI2(self):
        time_start = time.time()
        print('start')
        self.set_initial_value() # 初始化参数
        self.rolling_optimization() # 滚动优化
        time_end = time.time()
        print('end')
        print('total time', time_end - time_start)
        if self.save_data:
            self.data_record()
    def set_initial_value(self):
        '''初始化参数'''
        self.K[0] = 10
        self.K[1] = 10
        self.K[2] = 10
        '''初始化方差'''
        self.sigma[0] = 1.0
        self.sigma[1] = 0.3
        self.sigma[2] = 0.1
        '''初始化记录参数'''
        self.current_roll = 0
        self.current_training = 0
        '''初始化滚动环境'''
        self.env.reset()
    """ -----------------------------------------------------------计算轨迹回报,用于并行------------------------------------------------------------"""
    @jit(forceobj=True,nopython=True,nogil=True)
    def get_trajectory_loss(self, j):
        self.current_roll = j
        delta1 = np.random.normal(0, self.sigma[0], 1)
        delta2 = np.random.normal(0, self.sigma[1], 1)
        delta3 = np.random.normal(0, self.sigma[2], 1)
        cur_k1 = self.K[0] + delta1
        cur_k2 = self.K[1] + delta2
        cur_k3 = self.K[2] + delta3
        loss = self.reward_model.get_epsolid_reward(self.env,cur_k1, cur_k2, cur_k3)
        '''样本筛选'''
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
        multi_res = [poll.apply_async(self.get_trajectory_loss, (j,)) for j in range(roll_outs)]
        for j, res in enumerate(multi_res):
            self.k_delta[0, j] = res.get()[0]
            self.k_delta[1, j] = res.get()[1]
            self.k_delta[2, j] = res.get()[2]
            self.K_roll[0, j] = res.get()[3]
            self.K_roll[1, j] = res.get()[4]
            self.K_roll[2, j] = res.get()[5]
            self.loss[j] = res.get()[6]
    """ -----------------------------------------------------------策略改善------------------------------------------------------------"""
    def policy_improve(self):
        exponential_value_loss = np.zeros((roll_outs, 1), dtype=np.float64)  #
        probability_weighting = np.zeros((roll_outs, 1), dtype=np.float64)  # probability weighting of each roll
        if math.fabs(self.loss.max() -self.loss.min()) <= 1e-6: # avoid same loss vector
            for i2 in range(roll_outs):
                probability_weighting[i2] = 1 / roll_outs
        else:
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
        if flag1 < 1e-6 and flag2 < 1e-3:
            return True
        else :
            return False
    """ ----------------------------------------------------------策略迭代部分------------------------------------------------------------ """
    def training(self):
        i =0
        while i < training_times:
            # 分阶段优化 首先调整到局部最优 然后找到带有约束的满意解
            self.current_training = i
            # 方差衰减和可视化
            if self.current_training % self.attenuation_step_length == 0  and self.current_training != 0:
                self.sigma = self.sigma / self.alpha  # attenuation
                if self.current_training %3 == 0:
                    self.reward_model.model_simulation(self.K[0],self.K[1],self.K[2],1000)
                    plt.plot(self.loss_after_training[self.current_training - 3:self.current_training])
                    plt.title("loss between %d and %d epoch"%(self.current_training - 3,self.current_training))
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
            # 输出当前训练花费时间
            if(self.current_training+1 % self.attenuation_step_length == 0 ):
                print("!!!!!!!!!!!!! TIME !!!!!!!!",self.current_training,time.time()-first_time)
            i+=1
            print(i)
        if self.plot_photo and self.current_training !=0:
            plt.plot(self.K_after_training[0][:self.current_training],label="KP")
            plt.plot(self.K_after_training[1][:self.current_training],label="KD")
            plt.plot(self.K_after_training[2][:self.current_training],label="KI")
            plt.legend(loc="best")
            plt.savefig("K.png")
            plt.show()
            plt.plot(self.loss_after_training[:self.current_training])
            plt.savefig("loss.png")
            plt.show()
        return self.K[0],self.K[1],self.K[2],self.current_training+1
    def rolling_optimization(self,rolling_time=20,total_step=200):
        opt_times = int(total_step/rolling_time)
        theta_list = []
        theta_desired_list =[]
        alpha_list =[]
        delta_z_list = []
        iterator = 0
        self.K_after_roll_step[:, iterator] = self.K[:, 0]
        total_time = 0
        while iterator < opt_times :
            # print("test env 1",self.env.state[1])
            k = self.training()
            total_time = total_time + k[3]
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
        plt.plot(theta_list, 'b', label="time-theta",linewidth=linewidth)
        plt.plot(theta_desired_list, 'r', label="time-desired_theta",linewidth=linewidth)
        plt.legend(loc="best")
        plt.title("Rolling optimization graph After %d Opt" % iterator)
        plt.show()
        label = ["Kp", "Kd", "Ki"]
        color = ["r", "g", "b", "k"]
        line_style = ["-", "--", ":", "-."]
        marker = ['*', '^', 'h']
        "绘制K曲线"
        plt.figure(figsize=(2.8, 1.7), dpi=300)
        plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
        plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
        plt.xlabel("Number of Rolling Optimization",fontproperties='Times New Roman',fontsize=fontsize)
        plt.ylabel("$\mathcal{K}^{*}$ Value",fontproperties='Times New Roman',fontsize=fontsize)
        for i in range(3):
            plt.plot(self.K_after_roll_step[i][:iterator+1], label=label[i], color=color[i],
                     linestyle=line_style[i], marker=marker[i],linewidth=linewidth,markersize=markersize)
        plt.legend(loc='best',prop={'family': 'Times New Roman', 'size': legend_font_size})
        save_figure("./photo/exp1/", "K_Rolling_interval_%d.pdf"%rolling_time)
        plt.show()
        print(total_time)
        return alpha_list,delta_z_list,theta_list,theta_desired_list
'''----------------------------------------------------------------------主函数板块------------------------------'''
if __name__ == "__main__":
    ## 老师的路径积分
    first_time =time.time()
    poll = mp.Pool(mp.cpu_count())
    model = RL_PI2()
    reward_model = PID_model()
    alpha_list =[]
    delta_z_list = []
    theta_list = []
    theta_desire_list =[]
    # ## 第1组优化10次
    # model.set_initial_value()
    # alpha, delta_z, theta, theta_desired = model.rolling_optimization(rolling_time=100, total_step=800)
    # alpha_list.append(alpha)
    # delta_z_list.append(delta_z)
    # theta_list.append(theta)
    # theta_desire_list.append(theta_desired)
    ## 第2组仅仅优化一次
    model.set_initial_value()
    alpha, delta_z, theta, theta_desired = model.rolling_optimization(rolling_time=8000, total_step=8000)
    alpha_list.append(alpha)
    delta_z_list.append(delta_z)
    theta_list.append(theta)
    theta_desire_list.append(theta_desired)
    # ## 第3组对照组
    # alpha, delta_z, theta, theta_desired = reward_model.model_simulation(1.5,2.5,0.5,3000)
    # alpha_list.append(alpha)
    # delta_z_list.append(delta_z)
    # theta_list.append(theta)
    # theta_desire_list.append(theta_desired)
    # ## 第3组对照组
    # alpha, delta_z, theta, theta_desired = reward_model.model_simulation(1.5, 2.5, 0.5, 10000)
    # alpha_list.append(alpha)
    # delta_z_list.append(delta_z)
    # theta_list.append(theta)
    # theta_desire_list.append(theta_desired)
    ## 绘图
    plot_result(alpha_list,delta_z_list,theta_list,theta_desire_list,figure_number=1)
