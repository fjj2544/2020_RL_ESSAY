import numpy as np

class Planes_Env:
    # 初始化环境
    def __init__(self):
        # self.actions = [0,1]
        self.observation_dim = 4
        self.action_dim = 1
        self.action_bound = [-20,20]
        # 状态量分别为[飞机迎角 alpha, 飞机俯仰角theta, 飞机俯仰角速度q]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        self.max_steps = 400
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
        self.dtheta_desired = 0
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

    # 重设随机种子
    def seed(self, num_seed):
        np.random.seed(num_seed)

    # 还原环境
    # 输出：
    #   state(4;float)
    def reset(self, rand=False):
        n = np.random.randint(1, 1000, 1)
        np.random.seed(n)
        self.state = np.zeros((self.observation_dim, 1), dtype=np.float64)
        if rand:
            self.state[0:3] = np.random.uniform(0,3,size=(3,1))
        # if rand:
        #     self.state[0] = float(np.random.ranf(1)) * 10
        #     self.state[1] = float(np.random.ranf(1)) * 10
        #     self.state[2] = float(np.random.ranf(1)) * 10
        self.state[3] = self.theta_desired
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        return self.state

    # 按照动作前进一步
    # 输入：
    #   action(float)
    # 输出：
    #   state(4;float), reward(float), done(bool), info(None)
    def step(self, action):
        alpha, theta, q, theta_desired = self.state
        observation_pre = theta - self.theta_desired
        self.delta_z = float(np.clip(action,self.delta_z_threhold_min,self.delta_z_threhold_max))

        # 动力学方程 攻角alpha，俯仰角theta 俯仰角速度q  舵偏delta_z
        alpha_dot = q - self.b_alpha * alpha - self.b_delta_z * self.delta_z
        theta_dot = q
        q_dot = -self.a_alpha * alpha - self.a_q * q - self.a_delta_z * self.delta_z

        # 积分得到状态量
        q = q + self.tau * q_dot
        theta = theta + self.tau * theta_dot
        observation_cur = theta - self.theta_desired
        alpha = np.clip(alpha + self.tau * alpha_dot, self.alpha_threshold_min, self.alpha_threshold_max)
        # 更新状态
        self.steps_beyond_done += 1
        self.state = np.array([float(alpha), float(theta), float(q), float(self.theta_desired)])
        # 根据更新的状态判断是否结束
        # 回报函数
        self.reward = abs(theta - self.theta_desired)
        done = self.steps_beyond_done >= self.max_steps
        self.observation = np.array([observation_pre, observation_cur])
        info = None
        return self.state, self.reward, done, info

