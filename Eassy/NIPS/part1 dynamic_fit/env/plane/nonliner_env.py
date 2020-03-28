
import numpy as np
PITCH_D = 1.0   # 俯仰角目标【度】

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
        # 力
        self.Lift = 0.0
        self.Drag = 0.0
        self.Mz = 0.0
        self.temp = 0.0
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
        self.tau = 0.01

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
        # 力
        self.Lift = 0.0
        self.Drag = 0.0
        self.Mz = 0.0

        return self.observation

    # 采取动作
    def step(self, action):
        # 注意：ACTION的单位为度°
        action = action[0]
        # 限幅
        if action < -20:
            action = -20
        elif action > 20:
            action = 20
        # 变为度°
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
        self.temp = mz0
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
        self.Lift = Lift
        # 阻力
        Drag = Qdyn * CD * self.Sr
        self.Drag = Drag
        # 俯仰力矩
        Mz = Qdyn * mz * self.Sr * self.Lr
        self.Mz = Mz
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
        self.dtheta = (Thrust * np.sin(self.arfa) + Lift) / (self.mass * self.Mach * self.Vs) + np.cos(self.theta) * (self.Mach * self.Vs / (self.Re + self.altitude) - self.G0 / (self.Mach * self.Vs))
        # self.dtheta = (Thrust * np.sin(self.arfa) + Lift) / (self.mass * self.Mach * self.Vs) - np.cos(self.theta) * (self.G0 / (self.Mach * self.Vs))


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
        # 为神经网络添加数据


        self.observation = np.array([self.pitch*57.3, self.dpitch*57.3])
        return self.observation
