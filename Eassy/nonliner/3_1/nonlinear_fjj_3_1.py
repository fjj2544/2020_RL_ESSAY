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

PITCH_D_SIN_MAX = 5.0
PITCH_D_SIN_MIN = 1.0

# PID模块
Overshoot_target = 0.02/57.3    # 超参数目标
ts_target = 600                 # 目标调整时间
Static_error_target = 0.02/57.3 # 静态误差
adjust_bound = 0.02             # 波动限制
belief_times = 50               # 稳定次数

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
    plt.cla()

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
        self.pitch = 0 / 57.3
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
        # v变化[马赫/s]
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

        self.observation = np.array([self.pitch*57.3, self.dpitch*57.3,mz0,mz,Mz,self.Jz,Lift,Drag])
        return self.observation



'''跟踪目标函数'''
def target_func(cur_step):
    y = 1e-5/57.3
    return y
def target_func_dot(cur_step):
    y_dot = (math.pi* math.sin(cur_step * math.pi / 2000) )/ (57.3*2000)
    return y_dot
'''------------------------------PID控制器板块------------------------------'''

# ------------------------------PID模型------------------------------
#
# start_step -------> start_step + simulation_step -1
#
class PID_model():
    def __init__(self):
        self.env = Planes_Env()
    # 进行一次PID模拟
    def get_epsolid_reward(self,env,start_step,simulation_step = 5000, k1=1.5, k2=2.5, k3=0.5):
        self.env = copy.deepcopy(env)
        action_list = []
        ierror = 0
        if_alpha = False
        loss =0
        last_pitch_desire = 0
        alpha_count = 0
        action_count = 0
        # ----------进行循环控制----------
        for i in range(simulation_step):
            self.env.pitch_desired = target_func(cur_step=start_step+i)
            self.env.dpithch_desired = (self.env.pitch_desired - last_pitch_desire)/self.env.tau
            last_pitch_desire = self.env.pitch_desired

            error =  self.env.pitch_desired*57.3   - self.env.observation[0]
            derror = self.env.dpithch_desired*57.3 - self.env.observation[1]
            ierror = ierror + error * self.env.tau
            action = np.clip(k1 * error + k2 * derror + k3 * ierror, -20, 20)
            action_count += action**2
            action_list.append(action)
            loss += math.fabs(error)
            self.env.step(np.array([action]))
            if self.env.arfa*57.3 < -1 or self.env.arfa*57.3 >10:
                # if_alpha = True
                alpha_count += 1
        # if if_alpha:
        #     loss += 45000
        loss += alpha_count
        # loss += np.var(action_list)*simulation_step/2
        return loss
    ## 马赫数,高度,控制曲线,alpha曲线,theta曲线,jz ,Mz,mz0,jz
    def get_new_env(self, env,start_step,simulation_step = 5000,k1=1.5, k2=2.5, k3=0.5):
        self.env = copy.deepcopy(env)
        dez_list = []
        alpha = []
        theta = []
        desired_theta = []
        mh_list = []
        height_list = []
        q_list = []
        ierror = 0
        ierror_list = []
        derror_list = []
        error_list = []
        mz0_list = []
        mz_list = []
        Mz_list = []
        Jz_list = []
        lift_list = []
        Drag_list = []
        last_pitch_desire = 0
        for i in range(simulation_step):
            self.env.pitch_desired = target_func(cur_step=start_step + i)
            self.env.dpithch_desired = (self.env.pitch_desired - last_pitch_desire)/self.env.tau
            last_pitch_desire = self.env.pitch_desired
            error = (self.env.pitch_desired * 57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired * 57.3 - self.env.observation[1]
            print(self.env.dpithch_desired * 57.3 , self.env.observation[1])
            ierror = ierror + error * self.env.tau
            # 在当前情况下进行进行控制
            action = np.clip(k1 * error + k2 * derror + k3 * ierror,-20,20)
            self.env.step(np.array([action]))
            # 记录数据
            dez_list.append(action)
            alpha.append(self.env.arfa*57.3)
            theta.append(self.env.observation[0])
            q_list.append(self.env.observation[1])
            mz0_list.append(self.env.observation[2])
            mz_list.append(self.env.observation[3])
            Mz_list.append(self.env.observation[4])
            Jz_list.append(self.env.observation[5])
            lift_list.append(self.env.observation[6])
            Drag_list.append(self.env.observation[7])

            desired_theta.append(self.env.pitch_desired * 57.3)
            height_list.append(self.env.altitude)
            mh_list.append(self.env.Mach)
            error_list.append(error)
            derror_list.append(derror)
            ierror_list.append(ierror)

        return self.env,alpha, dez_list, theta, desired_theta,q_list,height_list,mh_list,error_list,ierror_list,derror_list,mz0_list,mz_list,Mz_list,Jz_list,lift_list,Drag_list

    def model_simulation( self,env,start_step,simulation_step = 5000,k1=1.5, k2=2.5, k3=0.5):
        self.env = copy.deepcopy(env)
        alpha = []
        theta = []
        desired_theta = []
        dez_list = []
        ierror = 0
        time = []
        last_pitch_desire = 0

        for i in range(simulation_step):
            # TODO:设计目标函数
            self.env.pitch_desired = target_func(cur_step=start_step + i)
            self.env.dpithch_desired = (self.env.pitch_desired - last_pitch_desire)/self.env.tau
            last_pitch_desire = self.env.pitch_desired

            error = (self.env.pitch_desired * 57.3 - self.env.observation[0])
            derror = self.env.dpithch_desired * 57.3 - self.env.observation[1]
            ierror = ierror + error * self.env.tau
            action = np.clip(k1 * error + k2 * derror + k3 * ierror, -20, 20)
            dez_list.append(action)
            self.env.step(np.array([action]))
            alpha.append(self.env.arfa * 57.3)
            theta.append(self.env.observation[0])
            desired_theta.append(self.env.pitch_desired * 57.3)
            time.append(i)
        plt.figure(2)
        plt.plot(time, theta, label="time-theta")
        plt.plot(time, desired_theta, label="time-desired_theta")
        plt.legend(loc="best")
        plt.title("Theta cruve between %d and %d"%(start_step,start_step+simulation_step))
        plt.show()
        return alpha, dez_list, theta, desired_theta

'''------------------------------FR-PI2模块------------------------------'''

""" -----------------------------------------------------------随机初始化参数-------------------------------------------------------------"""
Ki_Min = 0
Ki_Max = 100.0
Kp_Min = 0
Kp_Max = 100.0
Kd_Min = 0
Kd_Max = 100.0
""" -----------------------------------------------------------强化学习部分-------------------------------------------------------------"""
class RL_PI2:
    # 初始化
    def __init__(self,if_filter=RAND_INIT, attenuation_step_length=1, alpha=0.85):
        ## 滚动优化,用于记录当前阶段
        self.env = Planes_Env()
        # 记录全局最优K
        self.k_opt_record = np.zeros((3, 1))
        # 记录全局最优loss（与k对应)
        self.loss_opt_record = 1e10
        # 记录当前的loss
        self.cur_loss = 1e10
        # 记录训练上一轮loss
        self.last_loss = 1e10
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
        ## 下面属于滚动优化初始化
        self.opt_steps = 5000
        self.cur_step = 0
        # 初始化滚动环境
        self.env.reset()
        # debug间隔
        self.debug_interval = 5
        ## 最优K记录
        self.K_opt_after_training = np.zeros((3, training_times+Max_Adjust_times), dtype=np.float64)
        self.loss_opt_after_training = np.zeros((training_times + Max_Adjust_times, 1), dtype=np.float64)
        ## 关键
    ## 重新设置training 参数
    def reset_training_state(self):
        # 方差重置
        self.sigma[0] = 1.0
        self.sigma[1] = 0.3
        self.sigma[2] = 0.1
        # 记录参数重置
        self.current_roll = 0
        self.current_training = 0
        # 记录全局最优K,每一局应该刷新
        # 记录最初K和最初loss
        self.k_opt_record = self.K
        self.K_opt_after_training = np.zeros((3, training_times+Max_Adjust_times), dtype=np.float64)
        self.loss_after_training = np.zeros((training_times + Max_Adjust_times, 1), dtype=np.float64)
        self.K_after_training = np.zeros((3, training_times + Max_Adjust_times), dtype=np.float64)
        self.loss_opt_after_training = np.zeros((training_times + Max_Adjust_times, 1), dtype=np.float64)

        self.cur_loss = self.reward_model.get_epsolid_reward(self.env, self.cur_step, self.opt_steps, self.K[0], self.K[1], self.K[2])
        self.loss_opt_record = self.cur_loss
        self.last_loss= self.cur_loss

        self.K_after_training[:, self.current_training] = self.K[:, 0]
        self.loss_after_training[self.current_training] = self.cur_loss
        self.K_opt_after_training[:,self.current_training] = self.k_opt_record[:,0]
        self.loss_opt_after_training[self.current_training] = self.cur_loss
    def set_initial_value(self,INIT_K=[1.5,2.5,0.5]):
        if self.random_init:
            self.K[0] = random.uniform(Kp_Min,Kp_Max)
            self.K[1] = random.uniform(Kd_Min,Kd_Max)
            self.K[2] = random.uniform(Ki_Min,Ki_Max)
            self.K0 = copy.deepcopy(self.K)  # 记录初始值的
        else:
            self.K[0] = INIT_K[0]
            self.K[1] = INIT_K[1]
            self.K[2] = INIT_K[2]
            self.K0 = copy.deepcopy(self.K)  # 记录初始值的
        self.reset_training_state()
    def cal_trajectory_loss(self, j):
        self.current_roll = j
        delta1 = np.random.normal(0, self.sigma[0], 1)
        delta2 = np.random.normal(0, self.sigma[1], 1)
        delta3 = np.random.normal(0, self.sigma[2], 1)
        cur_k1 = self.K[0] + delta1
        cur_k2 = self.K[1] + delta2
        cur_k3 = self.K[2] + delta3
        cur_loss = self.reward_model.get_epsolid_reward(self.env, self.cur_step, self.opt_steps, cur_k1, cur_k2, cur_k3)
        return delta1,delta2,delta3,cur_k1,cur_k2,cur_k3, cur_loss
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
    def policy_improve(self):
        exponential_value_loss = np.zeros((roll_outs, 1), dtype=np.float64)
        probability_weighting = np.zeros((roll_outs, 1), dtype=np.float64)
        loss =copy.deepcopy(self.loss)
        if ((loss.max() - loss.min()) <1e-3):
            for i2 in range(roll_outs):
                probability_weighting[i2] = 1/roll_outs
        else:
            for i2 in range(roll_outs):
                exponential_value_loss[i2] = np.exp(-self.PI2_coefficient * (loss[i2] - loss.min())
                                                    / (loss.max() - loss.min()))
            for i2 in range(roll_outs):
                probability_weighting[i2] = exponential_value_loss[i2] / np.sum(exponential_value_loss)
        temp_k = np.dot(self.k_delta, probability_weighting)
        self.K = self.K + temp_k
        self.cur_loss = self.reward_model.get_epsolid_reward(self.env, self.cur_step, self.opt_steps, self.K[0], self.K[1],
                                                              self.K[2])
        # 确保K_opl的最优性
        ## 记录样本中最小的
        min_loss_index = np.argmin(loss)
        if loss[min_loss_index] < self.loss_opt_record:
            print("搜索改进loss", self.current_training,self.loss_opt_record,loss[min_loss_index])
            self.k_opt_record[:,0] = self.K_roll[:, min_loss_index]
            self.loss_opt_record = loss[min_loss_index]
        ## 记录结果最小的
        if self.cur_loss < self.loss_opt_record:
            print("合成改进loss",self.current_training)
            self.k_opt_record = self.K
            self.loss_opt_record = self.cur_loss
        # 更新上一轮的loss
        self.last_loss = self.cur_loss
    # 判断迭代停止,如果是best k 主导的呢？
    def iterator_finished(self):
        # flag1 = sum((self.K_after_training[:, self.current_training - 1] - self.K_after_training[:, self.current_training]) ** 2)
        # flag2 = self.loss_after_training[self.current_training]
        # print(self.K_opt_after_training.shape)
        flag1 = sum( (self.K_opt_after_training[:,self.current_training]-self.K_opt_after_training[:,self.current_training-1]) **2)
        flag2 = self.loss_opt_after_training[self.current_training]
        if flag1 < 1e-6 and flag2 < 1e-2:
            return True
        else:
            return False
    def training(self):
        print("开始训练")
        self.reset_training_state()
        while self.current_training < training_times:

            self.current_training += 1
            if self.current_training % self.attenuation_step_length == 0 :
                self.sigma = self.sigma/self.alpha
            # --------------------模拟仿真--------------------
            # if self.current_training - self.debug_interval >= 0 and self.current_training % self.debug_interval == 0 :
                # # print("绘制损失函数曲线")
                # plt.plot(self.loss_opt_after_training[self.current_training -self.debug_interval:self.current_training])
                # plt.title("loss between %d and %d epoch" % (self.current_training - self.debug_interval, self.current_training))
                # plt.show()
                # # print("绘制仿真曲线")
                # self.reward_model.model_simulation(self.env, self.cur_step, self.opt_steps, self.k_opt_record[0],self.k_opt_record[1],self.k_opt_record[2])

            # --------------------策略评估和改进--------------------
            self.policy_evl()
            self.policy_improve()
            # --------------------记录结果数据--------------------
            self.K_after_training[:, self.current_training] = self.K[:, 0]
            self.loss_after_training[self.current_training] = self.cur_loss
            self.K_opt_after_training[:, self.current_training] = self.k_opt_record[:,0]
            self.loss_opt_after_training[self.current_training] = self.loss_opt_record

            # --------------------调试信息--------------------
            if(self.current_training % self.attenuation_step_length == 0 ):
                print(self.current_training,time.time()-first_time)
            if self.current_training % 10 == 0 and self.loss_opt_record < self.loss_after_training[self.current_training]:
                print("更新最优参数")
                self.K = self.k_opt_record
                self.loss_after_training[self.current_training] = self.loss_opt_record
            if self.iterator_finished():
                break
        ## 后续数据对齐,保证数据对齐 training_times + 1
        if self.loss_after_training[self.current_training] > self.loss_opt_after_training[self.current_training]:
            self.K = self.K_opt_after_training[:, self.current_training]
        for it in range(self.current_training,training_times+1):
            self.loss_after_training[it] = self.loss_after_training[self.current_training]
            self.K_after_training[:,it] = self.K_after_training[:,self.current_training]
            self.K_opt_after_training[:, it] =  self.K_opt_after_training[:,self.current_training]
        return self.K[0],self.K[1],self.K[2],self.K_after_training,self.loss_after_training,self.current_training
    # 滚动优化
    def rolling_optimization(self, rolling_interval=250, total_step=5000):
        opt_times = math.ceil(total_step / rolling_interval)
        self.opt_steps = rolling_interval*2  #往后优化多少步
        theta_list = []
        theta_desired_list =[]
        alpha_list =[]
        delta_z_list = []
        K_list  =[]
        loss_list = []
        mh_list = []
        height_list = []
        q_list = []
        error_list =[]
        ierror_list = []
        derror_list = []
        mz0_list = []
        mz_list = []
        Mz_list = []
        Jz_list = []
        lift_list = []
        Drag_list = []

        iterator = 0
        max_time = 0
        while iterator < opt_times:
            self.training()
            max_time = max_time+ self.current_training
            ## 迭代多少次和当前训练没有关系
            for i in range(self.current_training+1):
                K_list.append(self.K_after_training[:,i])
                loss_list.append(self.loss_after_training[i])
            self.env,alpha, delta_z, theta, theta_desired,q,height,mh,error,ierror,derror,mz0,mz,Mz,Jz,lift,drag = self.reward_model.get_new_env(self.env, self.cur_step, rolling_interval, self.K[0], self.K[1], self.K[2])
            self.FR_time[iterator] = self.current_training
            self.K_after_roll_step[:, iterator] = self.K[:, 0]
            iterator += 1
            self.cur_step = iterator * rolling_interval
            for item in alpha:
                alpha_list.append(item)
            for item in delta_z:
                delta_z_list.append(item)
            for item in theta:
                theta_list.append(item)
            for item in theta_desired:
                theta_desired_list.append(item)
            for item in mh:
                mh_list.append(item)
            for item in q:
                q_list.append(item)
            for item in height:
                height_list.append(item)
            for item in error:
                error_list.append(item)
            for item in ierror:
                ierror_list.append(item)
            for item in derror:
                derror_list.append(item)
            for item in mz0:
                mz0_list.append(item)
            for item in mz:
                mz_list.append(item)
            for item in Mz:
                Mz_list.append(item)
            for item in Jz:
                Jz_list.append(item)
            for item in lift:
                lift_list.append(item)
            for item in drag:
                Drag_list.append(item)
            if iterator % 1==0:
                plt.figure(2)
                plt.plot(theta_list, 'b+', label="time-pitch")
                plt.plot(theta_desired_list, 'r', label="time-desired_pitch")
                plt.legend(loc="best")
                plt.title("Rolling optimization graph After %d Opt" % iterator)
                plt.show()
                # plt.plot(delta_z_list, 'r+', label="time-action")
                # plt.legend(loc="best")
                # plt.title("Action Graph After %d Opt" % iterator)
                # plt.show()

        save_dir = "./photo/exp1/"

        ###############绘制theta 曲线
        plt.figure(2)
        plt.plot(theta_list, 'b+', label="time-pitch")
        plt.plot(theta_desired_list, 'r', label="time-desired_pitch")
        plt.legend(loc="best")
        plt.title("Rolling optimization graph After %d Opt" % iterator)
        save_figure(save_dir,"theta.png")
        # plt.show()
        ###############绘制马赫曲线
        plt.plot(mh_list, 'b+', label="mach")
        plt.legend(loc="best")
        plt.title("Mach Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"mach.png")
        ## 绘制高度曲线
        plt.plot(height_list, 'b+', label="height")
        plt.legend(loc="best")
        plt.title("height Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"height.png")
        ## 绘制alpha曲线
        plt.plot(alpha_list, 'b+', label="alpha")
        plt.legend(loc="best")
        plt.title("alpha Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"alpha.png")
        ## 绘制action曲线
        plt.plot(delta_z_list, 'b+', label="delta_z")
        plt.legend(loc="best")
        plt.title("delta_z Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"delta_z.png")
        # 绘制q曲线
        plt.plot(q_list, 'b+', label="q")
        plt.legend(loc="best")
        plt.title("q Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"q.png")
        # 绘制p曲线
        plt.plot(error_list, 'b+', label="error")
        plt.legend(loc="best")
        plt.title("error Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"error.png")
        # 绘制i曲线
        plt.plot(ierror_list, 'b+', label="ierror")
        plt.legend(loc="best")
        plt.title("ierror Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"ierror.png")

        # 绘制d曲线
        plt.plot(derror_list, 'b+', label="derror")
        plt.legend(loc="best")
        plt.title("derror Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"derror.png")

        # 绘制mz0曲线
        plt.plot(mz0_list, 'b+', label="mz0")
        plt.legend(loc="best")
        plt.title("mz0 Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"mz0.png")

        # 绘制mz曲线
        plt.plot(mz_list, 'b+', label="mz")
        plt.legend(loc="best")
        plt.title("mz Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"mz.png")

        # 绘制Mz曲线
        plt.plot(Mz_list, 'b+', label="Mz")
        plt.legend(loc="best")
        plt.title("Mz Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"Mz.png")

        # 绘制Jz曲线
        plt.plot(Jz_list, 'b+', label="Jz")
        plt.legend(loc="best")
        plt.title("Jz Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"Jz.png")

        # 绘制lift曲线
        plt.plot(lift_list, 'b+', label="lift")
        plt.legend(loc="best")
        plt.title("lift Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"lift.png")

        # 绘制drag曲线
        plt.plot(Drag_list, 'b+', label="drag")
        plt.legend(loc="best")
        plt.title("drag Graph After %d Opt" % iterator)
        # plt.show()
        save_figure(save_dir,"drag.png")
        ###############绘制K曲线
        label = ["Kp", "Kd", "Ki"]
        color = ["r", "g", "b", "k"]
        line_style = ["-", "--", ":", "-."]
        marker = ['*', '^', 'h']
        plt.xticks(fontproperties='Times New Roman')
        plt.yticks(fontproperties='Times New Roman')
        plt.xlabel("Number of Rolling Optimization")
        plt.ylabel("$\mathcal{K}^{*}$ Value")
        for i in range(3):
            plt.plot(self.K_after_roll_step[i][:iterator], label=label[i], color=color[i],
                     linestyle=line_style[i], marker=marker[i])
        plt.legend(loc='best', prop={'family': 'Times New Roman'})
        plt.title("$\mathcal{K}^{*}$ Iteration Graph", fontdict={'family': 'Times New Roman'})
        save_figure("./photo/exp1/", "K_Rolling_interval_%d_%d_%d_%d.png" % (rolling_interval, self.K0[0], self.K0[1], self.K0[2]))
        # plt.show()
        ## 绘制训练时间曲线
        plt.figure(figsize=(2.8, 1.7), dpi=300)
        plt.xticks(fontproperties='Times New Roman',fontsize=fontsize)
        plt.yticks(fontproperties='Times New Roman',fontsize=fontsize)
        plt.xlabel("Number of Rolling Optimization",fontproperties='Times New Roman', fontsize=fontsize)
        plt.ylabel("Number of iterations",fontproperties='Times New Roman', fontsize=fontsize)
        x_major_locator = plt.MultipleLocator(1)
        y_major_locator = plt.MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.plot(self.FR_time[:iterator], label='Number of iterations', color=color[0],
                 linestyle=line_style[0], marker='^', markersize=4.5, linewidth=linewidth)
        plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
        save_figure("./photo/exp1/","time.png" )
        # plt.show()
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



    training_times = 20
    model = RL_PI2(if_filter=False, attenuation_step_length=1, alpha=0.85)
    model.set_initial_value([0, 0, 0])
    alpha, delta_z, theta, theta_desired, k_after, loss, cur_time = model.rolling_optimization(rolling_interval=250,
                                                                                               total_step=4000)
    alpha_list.append(alpha)
    delta_z_list.append(delta_z)
    theta_list.append(theta)
    theta_desire_list.append(theta_desired)
    k_list.append(k_after)
    loss_list.append(loss)
    maxtime = max(maxtime, cur_time)
    training_times = maxtime

    # plot_result(alpha_list,delta_z_list,theta_list,theta_desire_list,figure_number=1)
    # plot_loss_k(k_list, loss_list, maxtime,figure_number=1)
